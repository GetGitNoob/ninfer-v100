#include "serve/serve_options.h"
#include "serve/translate.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == ninfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninfer::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          defaults.kv_capacity.explicit_tokens == defaults.max_context,
                      "default KV capacity does not follow max context");
    failures += check(defaults.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninfer::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(!defaults.default_reasoning_effort,
                      "reasoning effort is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninfer::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninfer-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string kv_help = serve_usage_text("ninfer-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninfer-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");

    const ServeOptions reasoning_low =
        parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "low"});
    failures += check(
        reasoning_low.default_reasoning_effort == RequestedReasoningEffort::Low,
        "--default-reasoning-effort low did not preserve its value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == ninfer::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninfer-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"ninfer-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == ninfer::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninfer-serve", "model.ninfer", "--device-state-slots", "3", "--host-state-slots",
               "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions response_store =
        parse({"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking                 = true;
    prompt_capabilities.reasoning_effort.low            = true;
    prompt_capabilities.reasoning_effort.xhigh          = true;
    prompt_capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    const auto semantics = resolve_prompt_semantics(request, defaults, prompt_capabilities);
    failures += check(!semantics.reasoning_effort &&
                          semantics.effective_reasoning_effort == ninfer::ReasoningEffort::XHigh,
                      "omitted reasoning effort did not resolve to the template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking =
        resolve_prompt_semantics(request, thinking_budget, prompt_capabilities);
    failures += check(!non_thinking.effective_reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults, prompt_capabilities);
    failures +=
        check(explicit_effort.reasoning_effort == ninfer::ReasoningEffort::Low &&
                  explicit_effort.effective_reasoning_effort == ninfer::ReasoningEffort::Low,
              "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    // --- Server default reasoning effort (--default-reasoning-effort) resolution ---
    // Capabilities that accept every effort and default to xhigh, mirroring the production
    // qwen3.8 chat template: the server default can only lower the effective effort.
    ninfer::PromptCapabilities effort_caps;
    effort_caps.enable_thinking                 = true;
    effort_caps.reasoning_effort.low            = true;
    effort_caps.reasoning_effort.medium         = true;
    effort_caps.reasoning_effort.xhigh          = true;
    effort_caps.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;

    const ServeOptions server_low    =
        parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "low"});
    const ServeOptions server_medium =
        parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "medium"});

    auto new_request = [] {
        GenerationRequest request;
        request.max_tokens = 1;
        return request;
    };

    // 1. No server default and no client value: the template default is preserved (xhigh).
    {
        const auto resolved = resolve_prompt_semantics(new_request(), defaults, effort_caps);
        failures += check(
            resolved.enable_thinking && !resolved.reasoning_effort &&
                resolved.effective_reasoning_effort == ninfer::ReasoningEffort::XHigh,
            "no default and no client effort did not preserve the template default");
    }
    // 2. Server default low and no client value: resolves to low.
    {
        const auto resolved = resolve_prompt_semantics(new_request(), server_low, effort_caps);
        failures += check(
            resolved.enable_thinking && resolved.reasoning_effort == ninfer::ReasoningEffort::Low &&
                resolved.effective_reasoning_effort == ninfer::ReasoningEffort::Low,
            "server default low without a client effort did not resolve to low");
    }
    // 3. Server default low and client medium: the client medium wins.
    {
        auto request = new_request();
        request.reasoning_effort = RequestedReasoningEffort::Medium;
        const auto resolved = resolve_prompt_semantics(request, server_low, effort_caps);
        failures += check(
            resolved.enable_thinking &&
                resolved.reasoning_effort == ninfer::ReasoningEffort::Medium,
            "client medium did not take precedence over server default low");
    }
    // 4. Server default low and client xhigh: the client xhigh wins.
    {
        auto request = new_request();
        request.reasoning_effort = RequestedReasoningEffort::XHigh;
        const auto resolved = resolve_prompt_semantics(request, server_low, effort_caps);
        failures += check(
            resolved.enable_thinking &&
                resolved.reasoning_effort == ninfer::ReasoningEffort::XHigh,
            "client xhigh did not take precedence over server default low");
    }
    // 5. Server default low and client none: thinking is disabled.
    {
        auto request = new_request();
        request.reasoning_effort = RequestedReasoningEffort::None;
        const auto resolved = resolve_prompt_semantics(request, server_low, effort_caps);
        failures += check(
            !resolved.enable_thinking && !resolved.reasoning_effort &&
                !resolved.effective_reasoning_effort,
            "client none with server default low did not disable thinking");
    }
    // 6. Server default medium and client low: the client low wins.
    {
        auto request = new_request();
        request.reasoning_effort = RequestedReasoningEffort::Low;
        const auto resolved = resolve_prompt_semantics(request, server_medium, effort_caps);
        failures += check(
            resolved.enable_thinking &&
                resolved.reasoning_effort == ninfer::ReasoningEffort::Low,
            "client low did not take precedence over server default medium");
    }
    // 7. --no-thinking conflicts with an incompatible server default, in either order;
    //    a server default of "none" stays compatible with --no-thinking.
    {
        bool low_before_no_thinking_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "low",
                         "--no-thinking"});
        } catch (const std::invalid_argument&) { low_before_no_thinking_rejected = true; }
        failures += check(low_before_no_thinking_rejected,
                          "--no-thinking after --default-reasoning-effort low was accepted");
        bool no_thinking_before_low_rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--no-thinking", "--default-reasoning-effort",
                         "low"});
        } catch (const std::invalid_argument&) { no_thinking_before_low_rejected = true; }
        failures += check(no_thinking_before_low_rejected,
                          "--default-reasoning-effort low after --no-thinking was accepted");
        const ServeOptions no_thinking_none =
            parse({"ninfer-serve", "model.ninfer", "--no-thinking", "--default-reasoning-effort",
                   "none"});
        failures += check(
            !no_thinking_none.enable_thinking &&
                no_thinking_none.default_reasoning_effort == RequestedReasoningEffort::None,
            "--no-thinking did not accept a compatible --default-reasoning-effort none");
    }
    // 8a. Assistant prefill: the server default must not force an effort onto a prefill, and a
    //     prefill that still has thinking enabled is rejected, exactly as before. A client
    //     opt-out of thinking keeps the prefill non-thinking.
    {
        auto request = new_request();
        request.continuation = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        const ApiError prefill_thinking =
            api_error([&] { (void)resolve_prompt_semantics(request, server_low, effort_caps); });
        failures += check(prefill_thinking.code == "assistant_prefill_not_supported",
                          "assistant prefill with server-default thinking was not rejected");
        auto opt_out = new_request();
        opt_out.continuation   = ninfer::PromptContinuationMode::ContinueFinalAssistant;
        opt_out.enable_thinking = false;
        const auto opt_out_resolved = resolve_prompt_semantics(opt_out, server_low, effort_caps);
        failures += check(
            !opt_out_resolved.enable_thinking && !opt_out_resolved.effective_reasoning_effort,
            "an explicit client no-thinking did not keep an assistant prefill non-thinking");
    }
    // 8b. A tools request (auto mode) coexists with the server default reasoning effort and the
    //     resolved effort reaches the prompt alongside the tool definitions.
    {
        auto request = new_request();
        request.messages.push_back(
            ChatTurn{.role = ninfer::ChatRole::User,
                     .content = {ContentPart{.kind = ContentKind::Text,
                                                                     .text = "what is the weather"}}});
        request.tool_choice.mode = ToolChoiceMode::Auto;
        request.tools.push_back(
            ToolDefinition{.name = "get_weather", .description = "look up the weather",
                           .input_schema_json = "{\"type\":\"object\"}"});
        const auto resolved = resolve_prompt_semantics(request, server_low, effort_caps);
        const auto input    = to_prompt_input(
            request, resolved,
            [](const ContentPart&) -> ninfer::OwnedMedia {
                throw std::logic_error("the tools test request carries no media");
            });
        failures += check(
            resolved.enable_thinking && resolved.reasoning_effort == ninfer::ReasoningEffort::Low &&
                input.options.reasoning_effort == ninfer::ReasoningEffort::Low &&
                input.options.tool_jsons.size() == 1,
            "server default low did not coexist with a tools request");
    }
    // Invalid CLI values for --default-reasoning-effort are rejected.
    {
        bool bad_value_rejected = false;
        try { (void)parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "ultra"});
        } catch (const std::invalid_argument&) { bad_value_rejected = true; }
        failures += check(bad_value_rejected, "invalid --default-reasoning-effort value was accepted");
        bool missing_value_rejected = false;
        try { (void)parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort"}); }
        catch (const std::invalid_argument&) { missing_value_rejected = true; }
        failures += check(missing_value_rejected, "value-less --default-reasoning-effort was accepted");
    }
    // Every supported effort value parses and preserves its enum.
    for (const char* value : {"none", "minimal", "low", "medium", "high", "xhigh", "max"}) {
        const auto parsed = parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort",
                                   value});
        failures += check(
            parsed.default_reasoning_effort.has_value(),
            (std::string("supported --default-reasoning-effort value was rejected: ") + value)
                .c_str());
    }

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures += check(serve_usage_text("ninfer-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninfer-serve").find("device-state=max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninfer-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninfer-serve").find("--default-reasoning-effort") !=
                          std::string::npos,
                      "serve help omits --default-reasoning-effort");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninfer-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

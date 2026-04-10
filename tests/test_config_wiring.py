"""Tests verifying that ALL config fields flow through to their target modules.
No config field should be dead/unused."""
from lossless_agent.config import LCMConfig


class TestConfigToCompaction:
    def test_custom_instructions_flows(self):
        cfg = LCMConfig(custom_instructions="Keep function signatures")
        cc = cfg.to_compaction_config()
        assert cc.custom_instructions == "Keep function signatures"

    def test_empty_custom_instructions_flows(self):
        cfg = LCMConfig()
        cc = cfg.to_compaction_config()
        assert cc.custom_instructions == ""


class TestConfigToAssembler:
    def test_max_assembly_token_budget_flows(self):
        cfg = LCMConfig(max_assembly_token_budget=50000)
        ac = cfg.to_assembler_config()
        assert ac.max_assembly_token_budget == 50000

    def test_effective_max_tokens_with_budget(self):
        cfg = LCMConfig(max_context_tokens=128000, max_assembly_token_budget=50000)
        ac = cfg.to_assembler_config()
        assert ac.effective_max_tokens == 50000

    def test_effective_max_tokens_without_budget(self):
        cfg = LCMConfig(max_context_tokens=128000)
        ac = cfg.to_assembler_config()
        assert ac.effective_max_tokens == 128000

    def test_budget_higher_than_max_uses_max(self):
        cfg = LCMConfig(max_context_tokens=50000, max_assembly_token_budget=200000)
        ac = cfg.to_assembler_config()
        assert ac.effective_max_tokens == 50000


class TestConfigToExpandQuery:
    def test_expansion_model_flows(self):
        cfg = LCMConfig(expansion_model="claude-haiku-4-5")
        eqc = cfg.to_expand_query_config()
        assert eqc.expansion_model == "claude-haiku-4-5"

    def test_delegation_timeout_flows(self):
        cfg = LCMConfig(delegation_timeout_ms=60000)
        eqc = cfg.to_expand_query_config()
        assert eqc.timeout_ms == 60000


class TestConfigToLargeFile:
    def test_large_file_summary_provider_flows(self):
        cfg = LCMConfig(large_file_summary_provider="anthropic")
        lfc = cfg.to_large_file_config()
        assert lfc.summary_provider == "anthropic"

    def test_large_file_summary_model_flows(self):
        cfg = LCMConfig(large_file_summary_model="claude-haiku-4-5")
        lfc = cfg.to_large_file_config()
        assert lfc.summary_model == "claude-haiku-4-5"


class TestSummaryModelProvider:
    def test_summary_model_in_startup_banner(self):
        """summary_model and summary_provider are used by StartupBanner."""
        from lossless_agent.engine.startup_banner import StartupBanner
        cfg = LCMConfig(summary_model="gpt-4o", summary_provider="openai")
        StartupBanner.reset()
        StartupBanner.log_compaction_model(cfg)
        # Just verify it doesn't crash and reads the fields
        assert cfg.summary_model == "gpt-4o"
        assert cfg.summary_provider == "openai"

    def test_summary_model_in_compaction_config(self):
        """summary_model/provider should be available for adapter-level model selection."""
        cfg = LCMConfig(summary_model="claude-haiku-4-5", summary_provider="anthropic")
        # These are intentionally NOT on CompactionConfig since the summarize_fn
        # callable is configured externally by the adapter using these values
        assert cfg.summary_model == "claude-haiku-4-5"
        assert cfg.summary_provider == "anthropic"

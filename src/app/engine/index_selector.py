"""Index-aware option selection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import pandas as pd

from src.contract_selector import filter_chain, pick_best_contract
from src.tradier import TradierNotConfiguredError, fetch_option_chain_cached

from .index_common import DEFAULT_INDEX_RATIOS, ETF_PROXIES
from .index_mode import GammaSnapshot, IndexDataHealth, IndexPlanner
from .options_select import rules_for_symbol, score_contract

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContractsDecision:
    source: str
    chain: Optional[pd.DataFrame]
    health: Optional[IndexDataHealth] = None
    proxy_snapshot: Optional[GammaSnapshot] = None
    fallback_note: Optional[str] = None
    execution_proxy: Optional[Dict[str, object]] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)


class IndexOptionSelector:
    """Select index contracts with Polygon → Tradier → ETF fallback logic."""

    def __init__(self, planner: Optional[IndexPlanner] = None) -> None:
        self._planner = planner or IndexPlanner()

    async def contract_decision(
        self,
        symbol: str,
        *,
        prefer_delta: float,
        style: str | None = None,
    ) -> Tuple[Optional[Dict[str, object]], ContractsDecision]:
        base = symbol.upper()
        decision = ContractsDecision(source="unavailable", chain=None)
        health: IndexDataHealth | None = None
        if style:
            decision.diagnostics["style"] = style
        try:
            health = await self._planner.feed_health(base)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Index feed health check failed for %s: %s", base, exc)
        decision.health = health

        index_rules = rules_for_symbol(base)

        # 1. Try Polygon index chain
        try:
            polygon_chain = await self._planner.polygon_index_chain(base)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Polygon chain fetch failed for %s: %s", base, exc)
            polygon_chain = pd.DataFrame()

        contract = self._pick_contract(polygon_chain, index_rules, prefer_delta)
        if contract is not None:
            decision.source = "INDEX_POLYGON"
            decision.chain = polygon_chain
            decision.diagnostics["contract_source"] = "polygon"
            score_contract(contract, prefer_delta=prefer_delta)
            return contract, decision

        # 2. Fallback to Tradier index chain
        try:
            tradier_chain = await self._planner.tradier_index_chain(base)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Tradier index chain fetch failed for %s: %s", base, exc)
            tradier_chain = pd.DataFrame()

        contract = self._pick_contract(tradier_chain, index_rules, prefer_delta)
        if contract is not None:
            decision.source = "INDEX_TRADIER"
            decision.chain = tradier_chain
            decision.diagnostics["contract_source"] = "tradier"
            score_contract(contract, prefer_delta=prefer_delta)
            return contract, decision

        # 3. ETF proxy fallback with gamma translation
        proxy_symbol = ETF_PROXIES.get(base)
        if not proxy_symbol:
            decision.source = "ETF_UNSUPPORTED"
            decision.fallback_note = "No ETF proxy configured."
            return None, decision

        snapshot = await self._planner.gamma_snapshot(base)
        decision.proxy_snapshot = snapshot
        note = self._fallback_note(health, proxy_symbol)
        decision.fallback_note = note

        try:
            proxy_chain = await fetch_option_chain_cached(proxy_symbol)
        except TradierNotConfiguredError:
            proxy_chain = pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("ETF proxy chain fetch failed for %s: %s", proxy_symbol, exc)
            proxy_chain = pd.DataFrame()

        rules = rules_for_symbol(proxy_symbol)
        contract = self._pick_contract(proxy_chain, rules, prefer_delta)
        if contract is None:
            decision.source = "ETF_PROXY_EMPTY"
            return None, decision

        translated_targets = None
        if snapshot:
            translated_targets = {
                "gamma": snapshot.gamma_current,
                "gamma_mean": snapshot.gamma_mean,
                "ratio": snapshot.spot_ratio,
            }
        contract_payload = contract.copy()
        contract_payload["proxy_for"] = base
        contract_payload["liquidity_source"] = "etf_proxy"
        score_contract(contract_payload, prefer_delta=prefer_delta)

        decision.source = "ETF_PROXY"
        decision.chain = proxy_chain
        spot_ratio = None
        if snapshot:
            spot_ratio = round(snapshot.spot_ratio, 6)
        elif base in DEFAULT_INDEX_RATIOS:
            spot_ratio = DEFAULT_INDEX_RATIOS.get(base)
        if spot_ratio is not None:
            try:
                spot_ratio = round(float(spot_ratio), 6)
            except (TypeError, ValueError):
                spot_ratio = None
        proxy_payload = {
            "symbol": proxy_symbol,
            "underlying_proxy": proxy_symbol,
            "spot_ratio": spot_ratio,
            "note": note,
        }
        if snapshot:
            proxy_payload["gamma"] = round(snapshot.gamma_current, 6)
        decision.execution_proxy = proxy_payload
        if translated_targets:
            decision.diagnostics["gamma"] = translated_targets
        return contract_payload, decision

    @staticmethod
    def _pick_contract(
        chain: pd.DataFrame,
        rules: Dict[str, object],
        prefer_delta: float,
    ) -> Optional[Dict[str, object]]:
        if chain is None or chain.empty:
            return None
        try:
            filtered = filter_chain(chain, rules)
        except Exception:
            filtered = pd.DataFrame()
        if filtered is None or filtered.empty:
            return None
        best = pick_best_contract(filtered, prefer_delta=prefer_delta)
        if best is None:
            return None
        if hasattr(best, "to_dict"):
            return dict(best.to_dict())
        return dict(best)

    @staticmethod
    def _fallback_note(health: IndexDataHealth | None, proxy_symbol: str) -> str:
        if not health:
            return f"Index feeds degraded; using {proxy_symbol} proxy. Plan levels remain in index space."
        polygon_status = next(iter(health.polygon.values()), None)
        tradier_status = next(iter(health.tradier.values()), None)
        reason_parts = []
        if polygon_status and polygon_status.status != "healthy":
            reason_parts.append(f"Polygon={polygon_status.status}")
        if tradier_status and tradier_status.status != "healthy":
            reason_parts.append(f"Tradier={tradier_status.status}")
        reason = ", ".join(reason_parts) if reason_parts else "index liquidity thin"
        return f"Index feeds degraded ({reason}); using {proxy_symbol} proxy. Plan levels remain in index space."


__all__ = ["IndexOptionSelector", "ContractsDecision"]

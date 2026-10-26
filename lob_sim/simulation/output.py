from __future__ import annotations
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import Side


class OutputWriter:
    """Writes simulation output to Parquet files under a structured directory."""

    @staticmethod
    def write(
        output_dir: str,
        scenario_name: str,
        scenario_idx: int,
        state_dfs: dict[str, pd.DataFrame],
        agents: list[BaseArchetype],
    ) -> str:
        """
        Write all output files for one scenario run.
        Returns the path to the scenario directory.
        """
        scenario_dir = os.path.join(output_dir, scenario_name, f"scenario_{scenario_idx:04d}")
        os.makedirs(scenario_dir, exist_ok=True)

        # Enrich agent_orders with archetype column
        archetype_map = {a.agent_id: a.__class__.__name__ for a in agents}

        for name, df in state_dfs.items():
            if df.empty:
                continue
            if name == "agent_orders" and "agent_id" in df.columns:
                df = df.copy()
                df["archetype"] = df["agent_id"].map(archetype_map).fillna("__seed__")
            path = os.path.join(scenario_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)

        # Compute and write agent P&L
        trade_df = state_dfs.get("trade_tape", pd.DataFrame())
        pnl_df = OutputWriter._compute_agent_pnl(trade_df, agents)
        if not pnl_df.empty:
            pnl_df.to_parquet(os.path.join(scenario_dir, "agent_pnl.parquet"), index=False)

        return scenario_dir

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_agent_pnl(
        trade_df: pd.DataFrame,
        agents: list[BaseArchetype],
    ) -> pd.DataFrame:
        if trade_df.empty:
            return pd.DataFrame()

        records = []
        for agent in agents:
            aid = agent.agent_id
            archetype = agent.__class__.__name__

            buys  = trade_df[trade_df["buyer_agent_id"]  == aid]
            sells = trade_df[trade_df["seller_agent_id"] == aid]

            total_bought_qty   = float(buys["quantity"].sum())
            total_sold_qty     = float(sells["quantity"].sum())
            total_bought_value = float((buys["price"] * buys["quantity"]).sum())
            total_sold_value   = float((sells["price"] * sells["quantity"]).sum())
            realized_pnl       = total_sold_value - total_bought_value

            # Fill-rate only for archetypes with a target quantity
            total_quantity = getattr(agent, "_total_quantity", None)
            side           = getattr(agent, "_side", None)
            fill_rate: float | None = None
            if total_quantity and total_quantity > 0:
                filled = total_bought_qty if side == Side.BUY else total_sold_qty
                fill_rate = filled / total_quantity

            records.append({
                "agent_id":         aid,
                "archetype":        archetype,
                "total_bought_qty": total_bought_qty,
                "total_sold_qty":   total_sold_qty,
                "realized_pnl":     realized_pnl,
                "fill_rate":        fill_rate,
            })

        return pd.DataFrame(records)

"""Inspectable Mesa 3 agents backed by the model's vectorized NumPy state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mesa

if TYPE_CHECKING:
    from regworld.abm.model import RegulationModel


class FirmAgent(mesa.Agent):
    """A thin Mesa view over one row of the vectorized firm state."""

    def __init__(self, model: RegulationModel, firm_id: int) -> None:
        super().__init__(model)
        self.firm_id = firm_id
        self.quarter = 0
        self.compliant = False
        self.alive = True
        self.revenue = 0.0
        self.audited = False
        self.fined = False
        self.profit_reward = 0.0
        self.step()

    def step(self) -> None:
        state = self.model.state
        i = self.firm_id
        self.quarter = state.quarter
        self.compliant = bool(state.y[i] > 0.5)
        self.alive = bool(state.alive[i])
        self.revenue = float(state.revenue[i])
        self.audited = bool(state.audited[i])
        self.fined = bool(state.fines[i] > 0.0)
        self.profit_reward = float(self.model.last_firm_rewards[i])


class SegmentAgent(mesa.Agent):
    def __init__(self, model: RegulationModel, segment_id: int) -> None:
        super().__init__(model)
        self.segment_id = segment_id
        self.trust = 0.0
        self.step()

    def step(self) -> None:
        self.trust = float(self.model.state.trust[self.segment_id])


class AssociationAgent(mesa.Agent):
    def __init__(self, model: RegulationModel, association_id: int) -> None:
        super().__init__(model)
        self.association_id = association_id
        self.enforcement_multiplier = 1.0

    def step(self) -> None:
        controls = self.model.last_strategic_controls
        if self.association_id < controls.association_enforcement_multiplier.size:
            self.enforcement_multiplier = float(
                controls.association_enforcement_multiplier[self.association_id]
            )


class RegulatorAgent(mesa.Agent):
    def __init__(self, model: RegulationModel) -> None:
        super().__init__(model)
        self.regulator_id = 0
        self.reward = 0.0

    def step(self) -> None:
        self.reward = self.model.last_regulator_reward

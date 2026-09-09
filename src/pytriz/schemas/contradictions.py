from pydantic import BaseModel, Field, computed_field


class ParameterPairSelection(BaseModel):
    improving_id: int = Field(description="ID of the TRIZ parameter being improved (beneficial effect)")
    preserving_id: int = Field(description="ID of the TRIZ parameter being worsened (negative side-effect)")


class PrincipleSelection(BaseModel):
    id: int = Field(description="ID of the best-matching TRIZ Inventive Principle from the candidates list")


class Principle(BaseModel):
    id: int
    name: str = Field(..., description="The name of the principle.")
    description: str = Field("", description="A brief description of the principle.")
    rules: list[str] = Field([], description="The rules to apply the principle.")
    hints: list[str] = Field([], description="Hints for applying the principle.")
    examples: list[str] = Field([], description="Examples of the principle in action.")

    @computed_field
    @property
    def text(self) -> str:
        parts = [self.name, self.description]
        if self.rules:
            parts.append("Rules: " + "; ".join(self.rules))
        if self.hints:
            parts.append("Hints: " + "; ".join(self.hints))
        # if self.examples:
        #    parts.append("Examples: " + "; ".join(self.examples))
        return "\n".join(parts)


class Separation(BaseModel):
    id: int
    name: str = Field(..., description="The name of the separation principle.")
    description: str = Field("", description="A brief description of the separation principle.")
    guidelines: list[str] = Field(default_factory=list, description="Guidelines for applying the separation principle.")
    principles: list[Principle] = Field(
        default_factory=list,
        description="Inventive principles recommended for applying this separation principle.",
    )

    @computed_field
    @property
    def text(self) -> str:
        parts = [self.name, self.description]
        if self.guidelines:
            parts.append("Guidelines: " + "; ".join(self.guidelines))
        if self.principles:
            parts.append("Recommended inventive principles: " + "; ".join(p.name for p in self.principles))
        return "\n".join(parts)


class Parameter(BaseModel):
    id: int
    name: str = Field(..., description="The name of the parameter.")
    description: str = Field("", description="A brief description of the parameter.")
    examples: list[str] = Field([], description="Examples of the parameter values.")

    @computed_field
    @property
    def text(self) -> str:
        return f"{self.name}: {self.description}"



class TCModel(BaseModel):
    action: str = Field(..., description="Concise description of the action.")
    positive_effect: str = Field(..., description="The improvement caused by the action.")
    negative_effect: str = Field(..., description="The deterioration caused by the action.")


class Contradictions(BaseModel):
    contradictions: list[TCModel]


class ContradictionResult(BaseModel):
    contradiction: TCModel
    improving_parameter: Parameter
    preserving_parameter: Parameter

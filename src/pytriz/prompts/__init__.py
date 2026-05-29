import logging
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import frontmatter
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATES_DIR = str(files("pytriz") / "prompts")


class TemplateMetadata(BaseModel):
    type: Literal["system", "developer", "user"] = "system"
    author: str = ""
    version: int = 1
    labels: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    json_schema: dict[str, Any] | None = None


class PromptModel(TemplateMetadata):
    name: str
    prompt: str

    def compile(self, **kwargs) -> str:
        env = Environment(
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = env.from_string(self.prompt)
        return template.render(**kwargs)


def _get_env(templates_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(templates_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _load_template(template_name: str, env: Environment) -> tuple[str, TemplateMetadata]:
    if env.loader is None:
        raise FileNotFoundError(f"No template loader configured for template: {template_name}")

    template_source, _, _ = env.loader.get_source(env, template_name)
    post = frontmatter.loads(template_source)
    metadata = cast(dict[str, Any], post.metadata)

    return post.content, TemplateMetadata(
        type=cast(Literal["system", "developer", "user"], metadata.get("type", "system")),
        author=str(metadata.get("author", "")),
        version=int(metadata.get("version", 1)),
        labels=list(metadata.get("labels", [])),
        tags=list(metadata.get("tags", [])),
        config=dict(metadata.get("config", {})),
    )


def get_prompt(name: str, templates_dir: str = _DEFAULT_TEMPLATES_DIR) -> PromptModel:
    env = _get_env(templates_dir)
    try:
        content, metadata = _load_template(f"{name}.j2", env)
    except Exception as e:
        raise FileNotFoundError(f"Template file not found: {name}. Error: {e}") from e

    return PromptModel(name=name, prompt=content, **metadata.model_dump())


def get_all_prompts(templates_dir: str = _DEFAULT_TEMPLATES_DIR) -> list[str]:
    env = _get_env(templates_dir)
    try:
        templates = env.list_templates(filter_func=lambda x: x.endswith(".j2"))
        return [Path(t).stem for t in templates]
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        return []

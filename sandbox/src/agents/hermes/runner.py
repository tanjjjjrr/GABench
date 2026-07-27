from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.grading import extract_usage_from_jsonl
from src.utils.docker_utils import (
    inject_lobster_workspace,
    inject_xxx_models,
    run_background,
    run_warmup,
    setup_workspace,
    start_container,
)

load_dotenv()

logger = logging.getLogger(__name__)


class xxxAgent(BaseAgent):
    def __init__(
        self,
        gateway_port: int,
        xxx_api_key: str = "",
        xxx_base_url: str = "xxx",
        image_model: str | None = None,
        image: str | None = None,
    ) -> None:
        self.gateway_port = gateway_port
        self.image = image
        self.xxx_api_key = xxx_api_key
        self.xxx_base_url = xxx_base_url
        self.image_model = image_model if image_model is not None else os.environ.get("xxx_IMAGE_MODEL", "").strip()

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return "/root/.xxx/agents/main/sessions/chat.jsonl"

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = float(spec.timeout_seconds)

        try:
            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            start_container(
                spec.task_id,
                exec_path,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
                image=self.image,
                env_kv={
                    "xxx_API_KEY": self.xxx_api_key,
                    "xxx_BASE_URL": self.xxx_base_url,
                },
                port_bindings=spec.task.get("port_bindings", None),
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            setup_workspace(spec.task_id, thinking=spec.thinking)
            # setup_skills(spec.task_id, spec.task.get("skills", ""), spec.task.get("skills_path", ""), container_skills_root="/root/.xxx/workspace/skills")
            run_warmup(spec.task_id, spec.task.get("warmup", ""))
            if spec.models_config:
                inject_xxx_models(spec.task_id, spec.models_config)

            self._set_model(spec.task_id, spec.model)
            self._inject_xxx_key(spec.task_id, spec.model)
            image_model = self.image_model or spec.model
            self._set_image_model(spec.task_id, image_model)
            self._disable_web_search(spec.task_id)

            gateway_proc = run_background(
                spec.task_id,
                bash_cmd=(
                    f"export xxx_API_KEY='{self.xxx_api_key}' && "
                    f"export xxx_BASE_URL='{self.xxx_base_url}' && "
                    f"xxx gateway"
                ),
                log_path=spec.output_dir / "gateway.log",
            )
            logger.info("[%s] Waiting for gateway to be ready (2s)...", spec.task_id)
            time.sleep(2)

            safe_prompt = spec.prompt.replace("'", "'\\''")
            start_time = time.perf_counter()
            agent_bash_cmd = (
                f"export xxx_API_KEY='{self.xxx_api_key}' && "
                f"export xxx_BASE_URL='{self.xxx_base_url}' && "
                f"xxx agent --session-id chat --timeout {spec.timeout_seconds} --message '{safe_prompt}'"
            )
            agent_proc = run_background(
                spec.task_id,
                bash_cmd=agent_bash_cmd,
                log_path=spec.output_dir / "agent.log",
            )

            logger.info("[%s] Waiting for agent to finish...", spec.task_id)
            try:
                agent_proc.wait(timeout=spec.timeout_seconds)
                elapsed_time = time.perf_counter() - start_time
                logger.info(
                    "[%s] Agent finished successfully, elapsed: %.2f seconds",
                    spec.task_id,
                    elapsed_time,
                )
            except subprocess.TimeoutExpired:
                logger.info("[%s] Agent timed out...", spec.task_id)
                elapsed_time = float(spec.timeout_seconds)
                agent_proc.kill()
                agent_proc.wait()

            logger.info("[%s] Agent exit code: %s", spec.task_id, agent_proc.returncode)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except Exception as exc:
            logger.error("[%s] Execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict:
        transcript_host = output_dir / "chat.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        r_cp = subprocess.run(
            ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(transcript_host)],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode == 0 and transcript_host.exists():
            usage = extract_usage_from_jsonl(transcript_host)
        else:
            logger.warning("[%s] Transcript copy failed: %s", task_id, r_cp.stderr.strip())
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "request_count": 0,
            }
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def _set_model(self, task_id: str, model: str) -> None:
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"xxx models set '{model}'"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Model setup failed")
        logger.info("[%s] Model set: %s", task_id, model)

    def _inject_xxx_key(self, task_id: str, model: str = "") -> None:
        if not self.xxx_api_key:
            return

        # Extract bare model ID from "xxx/model-name" format
        model_id = model.split("/", 1)[1] if "/" in model else model

        key_json = json.dumps(self.xxx_api_key)
        base_url_json = json.dumps(self.xxx_base_url.rstrip("/"))
        model_id_json = json.dumps(model_id)

        # Inject into xxx.json under models.providers using mode=merge
        # This overrides the built-in "xxx" provider with our custom endpoint
        py_cmd = (
            "import json,pathlib;"
            "p=pathlib.Path('/root/.xxx/xxx.json');"
            "c=json.loads(p.read_text()) if p.exists() else {};"
            "m=c.setdefault('models',{});"
            "m['mode']='merge';"
            "prov=m.setdefault('providers',{}).setdefault('xxx',{});"
            f"prov['baseUrl']={base_url_json};"
            f"prov['apiKey']={key_json};"
            "prov['api']='xxx-completions';"
            f"prov.setdefault('models',[]);"
            # Add/update the model entry
            "existing=[x for x in prov['models'] if x.get('id')!=" + model_id_json + "];"
            "existing.append({'id':" + model_id_json + ",'name':" + model_id_json + ",'input':['text','image'],'contextWindow':131072,'maxTokens':8192});"
            "prov['models']=existing;"
            "p.write_text(json.dumps(c,indent=2))"
        )
        r = subprocess.run(
            ["docker", "exec", task_id, "python3", "-c", py_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] Failed to configure models.providers: %s", task_id, r.stderr.strip())
        else:
            logger.info("[%s] Configured xxx provider → %s (model: %s)",
                        task_id, self.xxx_base_url, model_id)

    def _set_image_model(self, task_id: str, model: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"xxx config set agents.defaults.imageModel.primary '{model}'"],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] imageModel set: %s", task_id, model)

    def _disable_web_search(self, task_id: str) -> None:
        # Set enabled=false AND replace the apiKey env-var reference with a literal
        # dummy value so Gateway doesn't crash on [WEB_SEARCH_KEY_UNRESOLVED_NO_FALLBACK]
        py_cmd = (
            "import json,pathlib;"
            "p=pathlib.Path('/root/.xxx/xxx.json');"
            "c=json.loads(p.read_text()) if p.exists() else {};"
            "s=c.setdefault('tools',{}).setdefault('web',{}).setdefault('search',{});"
            "s['enabled']=False;"
            "s['apiKey']='disabled';"
            "p.write_text(json.dumps(c,indent=2))"
        )
        r = subprocess.run(
            ["docker", "exec", task_id, "python3", "-c", py_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] Failed to disable web search: %s", task_id, r.stderr.strip())
        else:
            logger.info("[%s] Web search disabled and apiKey set to literal dummy", task_id)

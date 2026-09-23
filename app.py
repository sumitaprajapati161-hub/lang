import os
import subprocess
import traceback
from typing import TypedDict, List, Optional

import google.generativeai as genai
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=API_KEY)

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY
)

llm = llm_flash


class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    testbench: Optional[str]
    report: Optional[str]


@tool
def run_verilog_code(code: str, testbench: str) -> str:
    """Compile and simulate Verilog using Icarus Verilog."""
    try:
        code = code.replace("```verilog", "").replace("```", "").strip()
        testbench = testbench.replace("```verilog", "").replace("```", "").strip()

        with open("design.v", "w") as f:
            f.write(code)

        with open("testbench.v", "w") as f:
            f.write(testbench)

        compile_process = subprocess.run(
            ["iverilog", "-o", "simulation.out", "design.v", "testbench.v"],
            capture_output=True,
            text=True
        )

        if compile_process.returncode != 0:
            return "VERILOG COMPILATION ERROR:\n\n" + compile_process.stderr

        simulation_process = subprocess.run(
            ["vvp", "simulation.out"],
            capture_output=True,
            text=True
        )

        if simulation_process.returncode != 0:
            return "VERILOG SIMULATION ERROR:\n\n" + simulation_process.stderr

        output = simulation_process.stdout.strip()
        if not output:
            output = "Simulation completed with no output."

        return "VERILOG SIMULATION SUCCESSFUL\n\nSIMULATION OUTPUT:\n" + output

    except Exception:
        return "Execution Error:\n" + traceback.format_exc()


@tool
def generate_verilog_testbench(task_description: str) -> str:
    """Generate a Verilog testbench for the coding task."""
    prompt = f"""
You are a Senior Verilog Verification Engineer.

Coding task:
{task_description}

Generate a complete Verilog testbench.

Requirements:
1. Use Verilog only.
2. Do not use Python.
3. Instantiate the DUT.
4. Use the correct DUT module name.
5. Declare all required inputs and outputs.
6. Generate a clock if required.
7. Apply normal test cases.
8. Apply edge cases.
9. Use $display to show results.
10. Use $finish to stop simulation.
11. Return ONLY the Verilog testbench.
12. Do not use markdown code fences.
"""
    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
        return "\n".join(parts)

    return str(content)


def task_input_node(state: CrewState):
    print("\n" + "=" * 60)
    print("--- NEW VERILOG TASK ---")

    user_task = input(
        "Enter the Verilog coding task (or type 'exit' to quit): "
    ).strip()

    if user_task.lower() == "exit":
        return {"next_step": "exit"}

    return {
        "messages": [HumanMessage(content=user_task)],
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):
    print("\n[Developer] Generating Verilog code...")

    task = state["messages"][-1].content

    dev_prompt = f"""
You are an expert Verilog RTL engineer.

User task:
{task}

Write a complete Verilog solution.

Requirements:
1. Use Verilog only.
2. Do not write Python.
3. Do not use SystemVerilog.
4. Make the design simulatable.
5. Use synthesizable RTL wherever possible.
6. Include a proper module.
7. Return ONLY Verilog code.
8. Do not include explanations.
9. Do not use markdown code fences.
"""

    response = llm_flash.invoke(dev_prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
        code_str = "\n".join(parts)
    else:
        code_str = str(content)

    print("\nGenerated Verilog Code:")
    print("-" * 60)
    print(code_str)
    print("-" * 60)

    return {"code": code_str}


def real_time_tester(state: CrewState):
    print("\n[Tester] Generating Verilog testbench...")

    task = state["messages"][-1].content

    testbench = generate_verilog_testbench.invoke(task)

    if isinstance(testbench, list):
        parts = []
        for item in testbench:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
        testbench_str = "\n".join(parts)
    else:
        testbench_str = str(testbench)

    print("\nGenerated Testbench:")
    print("-" * 60)
    print(testbench_str)
    print("-" * 60)

    print("\n[Tester] Compiling Verilog...")

    execution_result = run_verilog_code.invoke({
        "code": state["code"],
        "testbench": testbench_str
    })

    report = f"""
============================================================
VERILOG EXECUTION REPORT
============================================================

TASK:
{task}

------------------------------------------------------------
GENERATED VERILOG
------------------------------------------------------------

{state["code"]}

------------------------------------------------------------
GENERATED TESTBENCH
------------------------------------------------------------

{testbench_str}

------------------------------------------------------------
SIMULATION RESULT
------------------------------------------------------------

{execution_result}

============================================================
"""

    return {
        "testbench": testbench_str,
        "report": report
    }


def manager_decision_node(state: CrewState):
    print("\n" + "=" * 60)
    print("--- MANAGER DASHBOARD ---")
    print(state.get("report", "No report available."))
    print("=" * 60)

    while True:
        user_input = input("\nCommand (store / another): ").lower().strip()

        if user_input in ["store", "another"]:
            break

        print("Please enter 'store' or 'another'.")

    if user_input == "store":
        return {"next_step": "archiver"}

    return {"next_step": "task_input"}


def archiver_node(state: CrewState):
    print("\n[Archiver] Verilog task stored successfully.")

    with open("verilog_report.txt", "w") as f:
        f.write(state.get("report", "No report available."))

    if state.get("code"):
        with open("generated_design.v", "w") as f:
            f.write(state["code"])

    if state.get("testbench"):
        with open("generated_testbench.v", "w") as f:
            f.write(state["testbench"])

    print("Files saved:")
    print("  - generated_design.v")
    print("  - generated_testbench.v")
    print("  - verilog_report.txt")

    return {"next_step": "exit"}


rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("task_input", task_input_node)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_node("manager_decision", manager_decision_node)
rt_workflow.add_node("archiver", archiver_node)

rt_workflow.add_edge(START, "task_input")


def route_from_input(state):
    if state.get("next_step") == "exit":
        return END
    return "developer"


rt_workflow.add_conditional_edges("task_input", route_from_input)
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", "manager_decision")


def route_from_decision(state):
    if state.get("next_step") == "archiver":
        return "archiver"
    return "task_input"


rt_workflow.add_conditional_edges("manager_decision", route_from_decision)
rt_workflow.add_edge("archiver", END)

rt_app = rt_workflow.compile()

print("\nInteractive Verilog pipeline compiled and ready.")


if __name__ == "__main__":
    try:
        rt_app.invoke(
            {"messages": []},
            config={"recursion_limit": 50}
        )

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

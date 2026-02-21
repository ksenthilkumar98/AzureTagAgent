#To install in the environment
pip install langgraph langchain azure-identity azure-mgmt-resource

import logging
import os
import json
import azure.functions as func
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient, SubscriptionClient
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- CONFIGURATION ---
app = func.FunctionApp()

# --- 1. Define Tenant-Level State ---
class AgentState(TypedDict):
    # We no longer hardcode one RG; we build a list of targets globally
    resources_to_process: List[dict] 
    current_resource: Optional[dict]
    logs: List[str]

# --- 2. Define Tools ---

def get_llm():
    """Connects to Azure AI Foundry"""
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        temperature=0
    )

# --- 3. Updated Nodes for Tenant Level ---

def scan_tenant_resources(state: AgentState):
    """
    Node: SCAN
    Action: Iterates through all subscriptions and filters RGs by naming convention.
    """
    logging.info("--- 🌍 Starting Tenant-Wide Scan ---")
    credential = DefaultAzureCredential()
    sub_client = SubscriptionClient(credential)
    
    all_targets = []
    
    # 1. Loop through all accessible Subscriptions
    for sub in sub_client.subscriptions.list():
        sub_id = sub.subscription_id
        logging.info(f"Scanning Sub: {sub.display_name}")
        
        resource_client = ResourceManagementClient(credential, sub_id)
        
        # 2. Get all Resource Groups in this subscription
        for rg in resource_client.resource_groups.list():
            rg_name = rg.name.lower()
            
            # --- NAMING CONVENTION FILTER ---
            # We only enter RGs that look like 'prod', 'dev', or 'qa'
            if any(key in rg_name for key in ["prod", "dev", "qa", "prd"]):
                logging.info(f"Checking Matching RG: {rg.name}")
                
                # 3. Find resources missing the Environment tag
                resources = resource_client.resources.list_by_resource_group(rg.name)
                for r in resources:
                    tags = r.tags or {}
                    if "Environment" not in tags:
                        all_targets.append({
                            "id": r.id, 
                            "name": r.name, 
                            "type": r.type,
                            "rg_name": rg.name, # Carry the RG name for LLM context
                            "sub_id": sub_id    # Needed to update the resource later
                        })

    logging.info(f"--- Tenant Scan Complete. Found {len(all_targets)} targets. ---")
    return {"resources_to_process": all_targets}

def analyze_with_llm(state: AgentState):
    """
    Node: AI ANALYSIS
    Now uses BOTH Resource Name and Resource Group Name for better accuracy.
    """
    resource = state["current_resource"]
    if not resource: return {}

    llm = get_llm()
    
    # We give the AI context about the Resource Group it was found in
    prompt = f"""
    You are a Cloud Governance Agent. Determine the 'Environment' tag (Production, Dev, QA).
    
    Context:
    - Resource Name: {resource['name']}
    - Resource Group Name: {resource['rg_name']}
    - Resource Type: {resource['type']}

    Decision Logic:
    - If the Resource Group name suggests an environment, prioritize that.
    - Example: 'rg-prod-web' always means 'Production'.
    
    Return JSON ONLY: {{"Environment": "value"}}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        decision = json.loads(content)
        resource["suggested_tag"] = decision.get("Environment", "Review_Needed")
    except Exception as e:
        logging.error(f"AI Analysis failed: {e}")
        resource["suggested_tag"] = "Error"

    return {"current_resource": resource}

def apply_tag(state: AgentState):
    resource = state["current_resource"]
    if not resource or "suggested_tag" not in resource: return {}

    new_tag = resource["suggested_tag"]
    logging.info(f"--- 🏷️ Tagging {resource['name']} in {resource['rg_name']} as {new_tag} ---")
    
    # We use the sub_id stored during the scan to get the right client
    credential = DefaultAzureCredential()
    client = ResourceManagementClient(credential, resource["sub_id"])
    
    try:
        client.resources.begin_update_by_id(
            resource["id"],
            api_version="2021-04-01",
            parameters={"tags": {"Environment": new_tag}}
        ).wait()
    except Exception as e:
        logging.error(f"Tagging failed for {resource['name']}: {e}")

    return {"logs": [f"Tagged {resource['name']} in {resource['rg_name']} as {new_tag}"]}

# --- 4. Graph Construction (Remains mostly the same) ---

def selector_node(state: AgentState):
    queue = state.get("resources_to_process", [])
    if not queue: return {"current_resource": None}
    item = queue[0]
    return {"current_resource": item, "resources_to_process": queue[1:]}

def route_logic(state: AgentState):
    if state["current_resource"]:
        return "analyze_with_llm"
    elif len(state["resources_to_process"]) > 0:
        return "selector"
    else:
        return END

workflow = StateGraph(AgentState)
workflow.add_node("scan_tenant_resources", scan_tenant_resources)
workflow.add_node("selector", selector_node)
workflow.add_node("analyze_with_llm", analyze_with_llm)
workflow.add_node("apply_tag", apply_tag)

workflow.set_entry_point("scan_tenant_resources")
workflow.add_edge("scan_tenant_resources", "selector")
workflow.add_conditional_edges("selector", route_logic, {
    "analyze_with_llm": "analyze_with_llm",
    "selector": "selector",
    "END": END
})
workflow.add_edge("analyze_with_llm", "apply_tag")
workflow.add_edge("apply_tag", "selector")

app_graph = workflow.compile()

# --- 5. Timer Trigger ---

@app.timer_trigger(schedule="0 */10 * * * *", arg_name="myTimer", run_on_startup=False) 
def azure_auto_tagger(myTimer: func.TimerRequest) -> None:
    logging.info('Tenant-level tagging cycle started.')
    
    initial_state = {
        "resources_to_process": [],
        "current_resource": None,
        "logs": []
    }

    try:
        app_graph.invoke(initial_state)
    except Exception as e:
        logging.error(f"Tenant workflow failed: {e}")
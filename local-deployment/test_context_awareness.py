import time
import asyncio
import json
from darkcircuit_agent_modular import Darkcircuit_Agent
from context_integration import init_darkcircuit_context

async def test_context_awareness():
    print("\n==== Testing Context Awareness System ====\n")
    
    # Step 1: Create agent with context awareness
    print("Creating agent with context awareness...")
    agent = Darkcircuit_Agent(model_name="gpt-3.5-turbo")
    
    # Initialize context awareness system
    context_manager = init_darkcircuit_context(
        agent,
        session_id=f"test_session_{int(time.time())}",
        persistence_path="./context_data"
    )
    
    # Step 2: Test basic context retention with two related questions
    print("\nTesting basic context retention...")
    
    # First question
    print("\nQuestion 1: What is nmap and how do I use it for basic scanning?")
    response1 = ""
    async for event in agent.run_agent_streaming("What is nmap and how do I use it for basic scanning?"):
        if event.get("type") == "token":
            response1 += event.get("value", "")
    print(f"Response: {response1[:150]}...")
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Follow-up question that relies on context
    print("\nQuestion 2: What other scan types can I use besides the ones you mentioned?")
    response2 = ""
    async for event in agent.run_agent_streaming("What other scan types can I use besides the ones you mentioned?"):
        if event.get("type") == "token":
            response2 += event.get("value", "")
    print(f"Response: {response2[:150]}...")
    
    # Step 3: Test entity extraction
    print("\nTesting entity extraction...")
    
    # Extract entities from the memory
    if hasattr(context_manager, 'memory') and hasattr(context_manager.memory, 'get_entities'):
        entities = context_manager.memory.get_entities()
        print(f"Extracted entities: {json.dumps(entities[:3], indent=2)}")
    
    # Step 4: Test persistence
    print("\nTesting persistence...")
    
    # Save state
    save_result = context_manager.save_state()
    print(f"State saved to: {save_result}")
    
    # Create a new context manager and load the state
    print("Creating new context manager and loading saved state...")
    new_agent = Darkcircuit_Agent(model_name="gpt-3.5-turbo")
    new_context_manager = init_darkcircuit_context(
        new_agent,
        session_id=context_manager.session_id,
        persistence_path="./context_data"
    )
    
    # Test if entities were loaded properly
    if hasattr(new_context_manager, 'memory') and hasattr(new_context_manager.memory, 'get_entities'):
        loaded_entities = new_context_manager.memory.get_entities()
        print(f"Loaded entities: {len(loaded_entities)}")
    
    # Step 5: Test task tracking
    print("\nTesting task tracking...")
    
    # Create a task
    if hasattr(context_manager, 'task_tracker'):
        task_id = context_manager.create_task(
            title="Test reconnaissance of target system",
            task_type="reconnaissance",
            description="Use nmap to scan the target system and identify open ports",
            priority="high"
        )
        print(f"Created task with ID: {task_id}")
        
        # Start the task
        context_manager.start_task(task_id)
        print("Started task")
        
        # Complete the task
        context_manager.complete_task(task_id, {"result": "Found open ports: 22, 80, 443"})
        print("Completed task")
        
        # Get task summary
        if hasattr(context_manager.task_tracker, 'get_summary'):
            summary = context_manager.task_tracker.get_summary()
            print(f"Task summary: {json.dumps(summary, indent=2)}")
    
    print("\n==== Context Awareness Testing Complete ====\n")

if __name__ == "__main__":
    asyncio.run(test_context_awareness())

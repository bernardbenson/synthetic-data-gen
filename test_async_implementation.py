#!/usr/bin/env python3
"""
Test script to verify async implementation without requiring OpenAI library.
"""

import asyncio
import time
import random
from typing import List, Dict, Any

# Mock OpenAI client for testing
class MockAsyncOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    class ChatCompletions:
        def __init__(self, client):
            self.client = client
        
        async def create(self, model: str, messages: List[Dict], **kwargs):
            # Simulate API call delay
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Return mock response
            class MockResponse:
                def __init__(self):
                    self.choices = [MockChoice()]
            
            class MockChoice:
                def __init__(self):
                    self.message = MockMessage()
            
            class MockMessage:
                def __init__(self):
                    self.content = f"Mock generated content for {model} with {len(messages)} messages. This is a test response that would normally be generated by OpenAI's API. The content is long enough to simulate realistic response length and word count targets."
                
                def strip(self):
                    return self.content
            
            return MockResponse()
    
    @property
    def chat(self):
        return MockAsyncOpenAI.ChatCompletions(self)

# Simplified version of the async functions for testing
async def mock_generate_text_with_openai(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3, client: MockAsyncOpenAI = None) -> str:
    """Mock version of generate_text_with_openai for testing."""
    if client is None:
        raise ValueError("Client must be provided")
    
    response = await client.chat.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert astrophysicist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.8,
        top_p=0.9
    )
    
    return response.choices[0].message.content

async def mock_generate_single_sample(
    class_name: str,
    sample_id: int,
    model: str,
    client: MockAsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Mock version of generate_single_sample for testing."""
    async with semaphore:
        try:
            print(f"Generating sample {sample_id} for class '{class_name}'")
            
            # Mock prompt
            prompt = f"Generate content for {class_name}"
            
            # Generate synthetic text
            synthetic_text = await mock_generate_text_with_openai(prompt, model, client=client)
            
            # Create sample
            sample = {
                "link": f"https://synthetic-data-gen.example.com/sample_{sample_id}",
                "full_text": synthetic_text,
                "labels": [class_name]
            }
            
            return sample
            
        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            return None

async def test_async_generation():
    """Test async generation with mock data."""
    print("Testing async implementation...")
    
    # Mock data
    classes = ["Class A", "Class B", "Class C", "Class D", "Class E"]
    samples_per_class = 3
    max_concurrent = 5
    
    # Setup mock client
    client = MockAsyncOpenAI(api_key="test-key")
    
    # Create semaphore
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all samples
    tasks = []
    sample_id = 1
    for class_name in classes:
        for i in range(samples_per_class):
            task = mock_generate_single_sample(
                class_name=class_name,
                sample_id=sample_id,
                model="gpt-4o-mini",
                client=client,
                semaphore=semaphore
            )
            tasks.append(task)
            sample_id += 1
    
    print(f"Created {len(tasks)} tasks")
    
    # Time the execution
    start_time = time.time()
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Filter successful results
    successful_results = [
        result for result in results 
        if result is not None and not isinstance(result, Exception)
    ]
    
    print(f"Completed {len(successful_results)} samples successfully")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Average time per sample: {(end_time - start_time) / len(successful_results):.2f} seconds")
    
    # Calculate theoretical synchronous time
    avg_delay = 0.3  # average mock delay
    sync_time = len(tasks) * avg_delay
    speedup = sync_time / (end_time - start_time)
    
    print(f"Theoretical synchronous time: {sync_time:.2f} seconds")
    print(f"Speedup factor: {speedup:.1f}x")
    
    return successful_results

if __name__ == "__main__":
    # Run the test
    results = asyncio.run(test_async_generation())
    
    if results:
        print(f"\nSample result structure:")
        print(f"- Link: {results[0]['link']}")
        print(f"- Labels: {results[0]['labels']}")
        print(f"- Text length: {len(results[0]['full_text'])} characters")
        print(f"- Text preview: {results[0]['full_text'][:100]}...")
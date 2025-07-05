

# Module level imports
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

# Global logger setup  
logger = logging.getLogger("BrowserAgent")

# Convenience functions for easy integration
async def create_browser_agent(
    session_id: Optional[str] = None,
    multi_mcp: Any = None
) -> 'BrowserAgent':
    """Create and initialize a BrowserAgent instance"""
    return BrowserAgent(session_id=session_id, multi_mcp=multi_mcp)

async def execute_browser_task(
    task: Dict[str, Any],
    session_id: Optional[str] = None,
    return_to: Optional[str] = None,
    multi_mcp: Any = None
) -> Dict[str, Any]:
    """Convenience function to execute a single browser task"""
    agent = await create_browser_agent(session_id, multi_mcp)
    return await agent.execute_task(task, return_to)

__all__: List[str] = [
    "BrowserAgent",
    "create_browser_agent",
    "execute_browser_task"
]

class BrowserAgent:
    """Specialized browser agent with multi-directional routing capabilities"""
    
    def __init__(self, session_id: Optional[str] = None, multi_mcp: Any = None):
        """Initialize BrowserAgent with session tracking"""
        self.session_id = session_id or str(__import__('uuid').uuid4())
        self.multi_mcp = multi_mcp
        self.logger = logging.getLogger(__name__)
        
        # Form state tracking to prevent repetitive actions
        self.form_state = {
            'filled_fields': set(),  
            'attempted_actions': [], 
            'current_form_data': {},  
            'submission_attempted': False
        }
        
        # Element stability tracking to handle ID flickering
        self.element_stability = {
            'last_elements_snapshot': None,  
            'stable_element_map': {},       
            'element_history': [],           
            'last_scan_timestamp': None,    
            'scan_count': 0                 
        }
        
        # Routing configuration
        self.current_agent = "browserAgent"
        self.return_to: Optional[str] = None
        self.routing_stack: List[str] = []
        
        # Browser-specific configurations
        self.browser_context = {
            "structured_output": True,  
            "strict_mode": True,        
            "viewport_mode": "visible"  
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"BrowserAgent.{self.session_id}")
        
        if not multi_mcp:
            self.logger.warning("BrowserAgent initialized without MCP executor - browser operations will fail!")

    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Execute an MCP tool with proper error handling"""
        try:
            if not self.multi_mcp:
                raise ValueError("MCP executor not initialized")
            
            # Execute the tool directly through MultiMCP
            result = await self.multi_mcp.call_tool(tool_name, parameters)
            return result
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {e}")
            raise

    async def execute_task(self, task: Dict[str, Any], return_to: Optional[str] = None) -> Dict[str, Any]:
        """Execute a browser task and track success for form state management"""
        action = task.get("action")
        parameters = task.get("parameters", {})
        
        self.logger.info(f"Executing browser task: {action}")
        
        try:
            # Execute the original task logic
            if action in ["get_elements", "get_interactive_elements"]:
                result = await self._handle_get_elements_task(action, parameters)
            elif action in ["click_element_by_index", "click"]:
                result = await self._handle_click_task(parameters)
            elif action in ["input_text", "type"]:
                result = await self._handle_input_task(parameters)
            elif action in ["go_to_url", "navigate", "go_back"]:
                result = await self._handle_navigation_task(action, parameters)
            elif action in ["get_comprehensive_markdown", "get_page_content"]:
                result = await self._handle_extraction_task(action, parameters)
            elif action == "get_session_snapshot":
                result = await self._handle_session_snapshot_task(parameters)
            else:
                result = await self._handle_generic_task(action, parameters)
            
            # Track successful form actions
            if result.get("success", False):
                await self._track_successful_form_action(action, parameters, result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute browser task {action}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "action": action,
                "error": error_msg,
                "message": f"Error executing action {action}: {str(e)}"
            }

    async def _track_successful_form_action(self, action: str, parameters: Dict, result: Dict):
        """Track successful form actions to update form state"""
        try:
            if action == "input_text":
                # Determine what field was filled based on the text content
                text = parameters.get("text", "").lower()
                index = parameters.get("index")
                
                # Map text content to form purposes
                if "@" in text and "." in text:
                    self.form_state['filled_fields'].add('email')
                    self.logger.info("✅ Marked email field as completed")
                    
                elif any(keyword in text for keyword in ['school', 'ai', 'course']):
                    self.form_state['filled_fields'].add('course')
                    self.logger.info("✅ Marked course field as completed")
                    
                elif any(keyword in text for keyword in ['sudarshan', 'name']):
                    self.form_state['filled_fields'].add('name')
                    self.logger.info("✅ Marked name field as completed")
                    
                elif any(keyword in text for keyword in ['1998', 'feb', 'birth', 'date']):
                    self.form_state['filled_fields'].add('date_of_birth')
                    self.logger.info("✅ Marked date of birth field as completed")
            
            elif action == "click_element_by_index":
                # Check if this was a radio button or dropdown selection
                message = result.get("message", "").lower()
                
                if "yes" in message or "married" in message:
                    self.form_state['filled_fields'].add('married_yes')
                    self.logger.info("✅ Marked married status as completed")
                    
                elif any(keyword in message for keyword in ['eag', 'course', 'select']):
                    self.form_state['filled_fields'].add('course_selection')
                    self.logger.info("✅ Marked course selection as completed")
                    
                elif "submit" in message:
                    self.form_state['submission_attempted'] = True
                    self.logger.info("✅ Marked form submission as attempted")
            
        except Exception as e:
            self.logger.error(f"Failed to track successful form action: {e}")

    async def _handle_get_elements_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle element retrieval tasks with proper field access"""
        try:
            fixed_parameters = {
                "structured_output": True,
                "strict_mode": parameters.get("strict_mode", True),
                "visible_only": parameters.get("visible_only", True),
                "viewport_mode": parameters.get("viewport_mode", "visible")
            }
            
            result = await self._execute_mcp_tool("get_interactive_elements", fixed_parameters)
            # print(result)
            return {
                "success": True,
                "elements": result.content[0].text,
                "message": f"Found interactive elements",
                "agent": self.current_agent,
                "structured_output": True,
                "mcp_result": result
            }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get elements: {str(e)}",
                "agent": self.current_agent
            }

    async def _handle_click_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle click tasks with proper element ID handling"""
        try:
            element_id = parameters.get("index")
            text_to_find = parameters.get("text")
            
            if element_id is not None:
                result = await self._execute_mcp_tool("click_element_by_index", {"index": element_id})
                return {
                    "success": True,
                    "message": f"Clicked element with ID: {element_id}",
                    "element_id": element_id,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            elif text_to_find:
                # First get elements to find by text
                elements_result = await self._handle_get_elements_task("get_elements", {})
                if not elements_result.get("success"):
                    return elements_result
                
                # Find element with matching text
                import json
                elements_result = json.loads(elements_result.get("elements", "[]"))
                data = elements_result.get("nav", []) + elements_result.get("forms", [])
                for element in data:
                    if text_to_find.lower() in element.get("desc", "").lower():
                        result = await self._execute_mcp_tool("click_element_by_index", {"index": element["id"]})
                        return {
                            "success": True,
                            "message": f"Clicked element with text '{text_to_find}' (ID: {element['id']})",
                            "element_id": element["id"],
                            "agent": self.current_agent,
                            "mcp_result": result
                        }
                
                return {
                    "success": False,
                    "error": f"Element with text '{text_to_find}' not found",
                    "agent": self.current_agent
                }
            else:
                return {
                    "success": False,
                    "error": "No element ID provided",
                    "agent": self.current_agent
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to click element: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_input_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multiple input/text entry tasks"""
        try:
            element_id = parameters.get("index")
            text = parameters.get("text")
            
            self.logger.info(f"🔤 Input task parameters: index={element_id}, text='{text}'")
            
            if element_id is None or text is None:
                return {
                    "success": False,
                    "error": f"Missing parameters - index: {element_id}, text: {text}",
                    "agent": self.current_agent
                }
            
            result = await self._execute_mcp_tool("input_text", {"index": element_id, "text": text})
            
            # Log the result for debugging
            self.logger.info(f"🔤 MCP input_text result: {type(result)} - {getattr(result, 'content', 'No content')}")
            
            # Check if MCP call was successful (MCP results don't have isError)
            if hasattr(result, 'content') or not hasattr(result, 'error'):
                return {
                    "success": True,
                    "message": f"Entered '{text}' text in field with index: {element_id}",
                    "element_id": element_id,
                    "text": text,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"MCP tool execution failed: {getattr(result, 'error', 'Unknown error')}",
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to input text: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_navigation_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation tasks"""
        try:
            # Handle open_tab separately (no URL required)
            if action == "open_tab":
                result = await self._execute_mcp_tool("open_tab", {})
                return {
                    "success": True,
                    "message": "Opened new browser tab",
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            
            # For other navigation tasks, URL is required
            url = parameters.get("url")
            if not url:
                return {
                    "success": False,
                    "error": f"No URL provided for {action}",
                    "agent": self.current_agent
                }

            # Navigate to URL using appropriate action
            if action in ["go_to_url", "navigate_to_url"]:
                result = await self._execute_mcp_tool("go_to_url", {"url": url})
                return {
                    "success": True,
                    "message": f"Navigated to: {url}",
                    "url": url,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            elif action == "navigate":
                # Legacy action, use go_to_url
                result = await self._execute_mcp_tool("go_to_url", {"url": url})
                return {
                    "success": True,
                    "message": f"Navigated to: {url}",
                    "url": url,
                    "agent": self.current_agent,
                    "mcp_result": result
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown navigation action: {action}",
                    "agent": self.current_agent
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute {action}: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_extraction_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content extraction tasks"""
        try:
            format_type = parameters.get("format", "markdown")
            
            if format_type == "structure":
                result = await self._execute_mcp_tool("get_enhanced_page_structure", {})
            else:
                result = await self._execute_mcp_tool("get_comprehensive_markdown", {})
            
            self.logger.info(f"Extracted content in {format_type} format")
            self.logger.info(f"Extraction result: {result}")  # Log first 100 chars
            
            return {
                "success": True,
                "content": result.content[0].text,
                "format": format_type,
                "agent": self.current_agent,
                "mcp_result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract content: {str(e)}",
                "agent": self.current_agent
            }
    
    async def _handle_session_snapshot_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle session snapshot tasks with proper result handling"""
        try:
            result = await self._execute_mcp_tool("get_session_snapshot", parameters)
            return {
                "success": True,
                "message": "Session snapshot captured successfully",
                "snapshot": result.content[0].text if hasattr(result, 'content') and result.content else "",
                "agent": self.current_agent,
                "mcp_result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get session snapshot: {str(e)}",
                "agent": self.current_agent
            }

    async def _handle_generic_task(self, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic browser tasks"""
        try:
            result = await self._execute_mcp_tool(action, parameters)
            return {
                "success": True,
                "message": f"Executed browser action: {action}",
                "action": action,
                "parameters": parameters,
                "agent": self.current_agent,
                "mcp_result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute {action}: {str(e)}",
                "agent": self.current_agent
            }
    
    # Routing methods for multi-directional agent communication
    async def route_to_agent(self, target_agent: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to another agent and manage routing stack"""
        if not target_agent:
            return {
                "success": False,
                "error": "No target agent specified",
                "agent": self.current_agent
            }

        try:
            # Add current agent to routing stack
            self.routing_stack.append(self.current_agent)
            
            # For now, return a mock response since we can't import other agents
            if target_agent == "perception":
                result = {
                    "success": True,
                    "message": "Routed to perception agent",
                    "agent": "perception",
                    "routed_from": self.current_agent
                }
            elif target_agent == "decision":
                result = {
                    "success": True,
                    "message": "Routed to decision agent",
                    "agent": "decision",
                    "routed_from": self.current_agent
                }
            elif target_agent == "execution":
                result = {
                    "success": True,
                    "message": "Routed to execution agent",
                    "agent": "execution",
                    "routed_from": self.current_agent
                }
            else:
                result = {
                    "success": False,
                    "error": f"Unknown target agent: {target_agent}",
                    "agent": self.current_agent
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to route to {target_agent}: {str(e)}",
                "agent": self.current_agent
            }
    
    async def return_control(self) -> Optional[str]:
        """Return control to the previous agent in the routing stack"""
        if self.routing_stack:
            return self.routing_stack.pop()
        return self.return_to
    
    def get_routing_context(self) -> Dict[str, Any]:
        """Get current routing context for debugging/monitoring"""
        return {
            "current_agent": self.current_agent,
            "return_to": self.return_to,
            "routing_stack": list(self.routing_stack),  # Create a copy
            "session_id": self.session_id
        }
    
    async def execute_complete_workflow(self, query: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Execute a complete browser workflow with intelligent step-by-step decision making"""
        self.logger.info(f"🚀 Starting INTELLIGENT browser workflow for: {query}")
        
        # Initialize workflow state
        workflow_results = []
        original_query = query
        
        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"🔄 Workflow iteration {iteration}/{max_iterations}")
            
            try:
                # Use intelligent step-by-step workflow instead of static task generation
                iteration_results = await self._execute_intelligent_workflow(query, workflow_results)
                workflow_results.extend(iteration_results)
                
                # Check if goal is achieved using intelligent analysis
                if await self._is_goal_achieved_intelligent(original_query, workflow_results):
                    self.logger.info("🎯 Goal achieved through intelligent workflow")
                    break
                    
                # If not achieved and not final iteration, generate continuation strategy
                if iteration < max_iterations:
                    continuation_query = await self._generate_intelligent_continuation(
                        original_query, workflow_results
                    )
                    if continuation_query:
                        query = continuation_query
                        self.logger.info(f"🔄 Continuing with intelligent strategy: {continuation_query}")
                    else:
                        self.logger.info("🎯 No further intelligent actions needed")
                        break
                        
            except Exception as e:
                self.logger.error(f"❌ Iteration {iteration} failed: {e}")
                workflow_results.append({
                    "iteration": iteration,
                    "error": str(e),
                    "success": False
                })
                
        # Return comprehensive results
        return {
            "success": len([r for r in workflow_results if r.get("success", False)]) > 0,
            "results": workflow_results,
            "iterations": iteration,
            "original_query": original_query,
            "agent": self.current_agent
        }

    async def _execute_intelligent_workflow(self, query: str, previous_results: List[Dict]) -> List[Dict[str, Any]]:
        """Execute intelligent step-by-step workflow using LLM for decision making"""
        results = []
        max_steps = 10  # Maximum steps per iteration
        
        self.logger.info("🧠 Starting intelligent step-by-step workflow")
        
        for step in range(1, max_steps + 1):
            self.logger.info(f"🤖 Intelligent Step {step}/{max_steps}")
            
            # Get current page state
            current_state = await self._get_current_page_state()
            
            # Use LLM to decide next action based on query, state, and previous results
            next_action = await self._decide_next_action_intelligent(
                query, current_state, previous_results + results
            )
            
            if not next_action or next_action.get("action") == "goal_achieved":
                self.logger.info("🎯 LLM determined goal is achieved")
                break
                
            # Execute the intelligently chosen action
            self.logger.info(f"🔄 Executing intelligent action: {next_action.get('action')} with parameters: {next_action.get('parameters', {})}")
            action_result = await self.execute_task(next_action)
            action_result["step"] = step
            action_result["reasoning"] = next_action.get("reasoning", "")
            results.append(action_result)
            
            # If action failed, use LLM to decide recovery strategy
            if not action_result.get("success", False):
                recovery_action = await self._decide_recovery_action(
                    query, action_result, current_state
                )
                if recovery_action:
                    self.logger.info(f"🔧 Executing recovery action: {recovery_action.get('action')}")
                    recovery_result = await self.execute_task(recovery_action)
                    recovery_result["step"] = f"{step}_recovery"
                    recovery_result["reasoning"] = recovery_action.get("reasoning", "")
                    results.append(recovery_result)
                    
        return results

    async def _get_current_page_state(self) -> Dict[str, Any]:
        """Get comprehensive current page state with enhanced stability tracking"""
        try:
            # Get page snapshot
            snapshot_result = await self._execute_mcp_tool("get_session_snapshot", {})
            
            # Get interactive elements with stability enhancement
            current_timestamp = __import__('datetime').datetime.now()
            
            # Check if we can use cached elements to avoid ID flickering
            if await self._should_use_cached_elements(current_timestamp):
                self.logger.info("🔒 Using cached elements to maintain ID stability")
                elements_result = self.element_stability['last_elements_snapshot']
                stable_elements = self.element_stability['stable_element_map']
            else:
                # Perform new scan with stability tracking
                self.logger.info("🔍 Performing new element scan with stability tracking")
                elements_result = await self._execute_mcp_tool("get_interactive_elements", {
                    "structured_output": True,
                    "strict_mode": True,
                    "visible_only": True
                })
                
                # Update stability tracking
                await self._update_element_stability(elements_result, current_timestamp)
                
                # Create stable element mapping with enhanced persistence
                stable_elements = await self._create_enhanced_stable_mapping(elements_result)
            
            return {
                "snapshot": snapshot_result.content[0].text if hasattr(snapshot_result, 'content') else str(snapshot_result),
                "elements": elements_result.content[0].text if hasattr(elements_result, 'content') else str(elements_result),
                "stable_elements": stable_elements,
                "timestamp": current_timestamp.isoformat(),
                "scan_count": self.element_stability['scan_count'],
                "stability_info": {
                    "using_cache": self.element_stability['last_elements_snapshot'] is not None,
                    "last_scan": self.element_stability['last_scan_timestamp'],
                    "element_count": len(stable_elements.get('by_form_purpose', {}))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get page state: {e}")
            return {"error": str(e), "timestamp": __import__('datetime').datetime.now().isoformat()}

    async def _should_use_cached_elements(self, current_timestamp) -> bool:
        """Determine if we should use cached elements to maintain stability"""
        if not self.element_stability['last_elements_snapshot']:
            return False
        
        # If last scan was less than 3 seconds ago, use cache to prevent flickering
        if self.element_stability['last_scan_timestamp']:
            time_diff = (current_timestamp - self.element_stability['last_scan_timestamp']).total_seconds()
            if time_diff < 3.0:
                return True
        
        # If we have fewer than 3 scans, be more conservative with caching
        if self.element_stability['scan_count'] < 3:
            return True
        
        return False

    async def _update_element_stability(self, elements_result, timestamp):
        """Update element stability tracking with new scan results"""
        self.element_stability['last_elements_snapshot'] = elements_result
        self.element_stability['last_scan_timestamp'] = timestamp
        self.element_stability['scan_count'] += 1
        
        # Keep history of element changes (last 5 scans)
        current_elements = await self._extract_element_signatures(elements_result)
        self.element_stability['element_history'].append({
            'timestamp': timestamp,
            'scan_count': self.element_stability['scan_count'],
            'element_signatures': current_elements
        })
        
        # Keep only recent history
        if len(self.element_stability['element_history']) > 5:
            self.element_stability['element_history'] = self.element_stability['element_history'][-5:]
        
        # Log stability information
        if self.element_stability['scan_count'] > 1:
            self._analyze_element_stability(current_elements)

    async def _extract_element_signatures(self, elements_result) -> List[str]:
        """Extract stable signatures from elements for stability tracking"""
        try:
            import json
            import re
            
            if hasattr(elements_result, 'content'):
                elements_text = elements_result.content[0].text
            else:
                elements_text = str(elements_result)
            
            signatures = []
            
            # Parse elements to create signatures based on stable attributes
            if elements_text.startswith('{'):
                # JSON format
                try:
                    elements_data = json.loads(elements_text)
                    if 'interactive_elements' in elements_data:
                        for element in elements_data['interactive_elements']:
                            sig = self._create_element_signature(element)
                            if sig:
                                signatures.append(sig)
                except:
                    pass
            else:
                # Line format: [index]<tag attrs>text />
                for line in elements_text.split('\n'):
                    line = line.strip()
                    if line and line.startswith('[') and ']' in line:
                        match = re.match(r'\[(\d+)\]<(\w+)([^>]*)>([^<]*)', line)
                        if match:
                            index, tag, attrs_str, text = match.groups()
                            
                            # Parse attributes
                            attrs = {}
                            attr_matches = re.findall(r'(\w+)=[\'"]([^\'"]*)[\'"]', attrs_str)
                            for attr_name, attr_value in attr_matches:
                                attrs[attr_name] = attr_value
                            
                            element = {
                                'index': int(index),
                                'tag': tag,
                                'attributes': attrs,
                                'text': text.strip()
                            }
                            
                            sig = self._create_element_signature(element)
                            if sig:
                                signatures.append(sig)
            
            return signatures
            
        except Exception as e:
            self.logger.error(f"Failed to extract element signatures: {e}")
            return []

    def _create_element_signature(self, element) -> str:
        """Create a stable signature for an element based on immutable attributes"""
        tag = element.get('tag', '').lower()
        attrs = element.get('attributes', {})
        text = element.get('text', '').strip()
        
        # Build signature from stable attributes (not index!)
        sig_parts = [f"tag:{tag}"]
        
        # Add stable identifying attributes
        for attr in ['id', 'name', 'type', 'placeholder', 'aria-label', 'role']:
            value = attrs.get(attr)
            if value:
                sig_parts.append(f"{attr}:{value}")
        
        # Add text content if meaningful
        if text and len(text.strip()) > 0:
            # Normalize text for signature
            normalized_text = ' '.join(text.split())[:50]  # First 50 chars, normalized
            sig_parts.append(f"text:{normalized_text}")
        
        return "|".join(sig_parts)

    def _analyze_element_stability(self, current_signatures):
        """Analyze and log element stability changes"""
        if len(self.element_stability['element_history']) >= 2:
            previous_signatures = self.element_stability['element_history'][-2]['element_signatures']
            
            # Find differences
            added = set(current_signatures) - set(previous_signatures)
            removed = set(previous_signatures) - set(current_signatures)
            stable = set(current_signatures) & set(previous_signatures)
            
            stability_pct = len(stable) / max(len(current_signatures), len(previous_signatures), 1) * 100
            
            self.logger.info(f"🔍 Element Stability Analysis:")
            self.logger.info(f"   📊 Stability: {stability_pct:.1f}% ({len(stable)}/{len(current_signatures)} elements)")
            
            if added:
                self.logger.info(f"   ➕ Added: {len(added)} elements")
            if removed:
                self.logger.info(f"   ➖ Removed: {len(removed)} elements")
            
            # Warn about low stability
            if stability_pct < 80:
                self.logger.warning(f"⚠️  Low element stability detected! This may cause ID flickering.")

    async def _create_enhanced_stable_mapping(self, elements_result) -> Dict[str, Any]:
        """Create enhanced stable mapping with better persistence"""
        # Create the basic stable mapping
        stable_mapping = await self._create_stable_element_mapping(elements_result)
        
        # Enhance with persistent mapping from previous scans
        if self.element_stability['stable_element_map']:
            # Merge with previous stable mappings where possible
            stable_mapping = await self._merge_with_previous_mapping(stable_mapping)
        
        # Update persistent mapping
        self.element_stability['stable_element_map'] = stable_mapping
        
        return stable_mapping

    async def _create_stable_element_mapping(self, elements_result) -> Dict[str, Any]:
        """Create stable element mapping using semantic attributes instead of dynamic indices"""
        try:
            import json
            import re
            
            mapping = {
                'by_form_purpose': {},  # Map by semantic purpose (email, name, etc.)
                'by_attributes': {},    # Map by stable attributes (id, name, etc.)
                'by_text': {},         # Map by text content
                'by_type': {},         # Map by input type
                'metadata': {
                    'created_at': __import__('datetime').datetime.now().isoformat(),
                    'total_elements': 0,
                    'interactive_elements': 0
                }
            }
            
            if hasattr(elements_result, 'content'):
                elements_text = elements_result.content[0].text
            else:
                elements_text = str(elements_result)
            
            mapping['metadata']['total_elements'] = elements_text.count('[')
            
            # Parse elements based on format
            if elements_text.startswith('{'):
                # JSON format
                elements_data = json.loads(elements_text)
                if 'interactive_elements' in elements_data:
                    for element in elements_data['interactive_elements']:
                        self._add_element_to_stable_mapping(element, mapping)
            else:
                # Line format: [index]<tag attrs>text />
                for line in elements_text.split('\n'):
                    line = line.strip()
                    if line and line.startswith('[') and ']' in line:
                        element = self._parse_element_line(line)
                        if element:
                            self._add_element_to_stable_mapping(element, mapping)
            
            mapping['metadata']['interactive_elements'] = len(mapping['by_form_purpose'])
            
            self.logger.info(f"🗺️  Created stable mapping with {mapping['metadata']['interactive_elements']} elements")
            self.logger.debug(f"   Form purposes: {list(mapping['by_form_purpose'].keys())}")
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"Failed to create stable element mapping: {e}")
            return {'by_form_purpose': {}, 'by_attributes': {}, 'by_text': {}, 'by_type': {}}

    def _parse_element_line(self, line: str) -> Optional[Dict]:
        """Parse element from line format: [index]<tag attrs>text />"""
        try:
            import re
            
            # Match pattern: [index]<tag attrs>text />
            match = re.match(r'\[(\d+)\]<(\w+)([^>]*)>([^<]*)', line)
            if not match:
                return None
            
            index, tag, attrs_str, text = match.groups()
            
            # Parse attributes
            attributes = {}
            attr_matches = re.findall(r'(\w+)=[\'"]([^\'"]*)[\'"]', attrs_str)
            for attr_name, attr_value in attr_matches:
                attributes[attr_name] = attr_value
            
            return {
                'index': int(index),
                'tag': tag.lower(),
                'attributes': attributes,
                'text': text.strip()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to parse element line: {e}")
            return None

    def _add_element_to_stable_mapping(self, element: Dict, mapping: Dict):
        """Add element to stable mapping using multiple identification strategies"""
        try:
            index = element.get('index')
            tag = element.get('tag', '').lower()
            attrs = element.get('attributes', {})
            text = element.get('text', '').strip().lower()
            
            element_data = {
                'index': index,
                'tag': tag,
                'attributes': attrs,
                'text': element.get('text', '').strip(),  # Keep original text case
                'signature': self._create_element_signature(element)
            }
            
            # 1. Map by form purpose using semantic analysis
            purpose = self._determine_form_purpose(element, text, attrs)
            if purpose:
                mapping['by_form_purpose'][purpose] = element_data
                self.logger.debug(f"   📌 Mapped {purpose} → index {index}")
            
            # 2. Map by stable attributes
            for attr_name in ['id', 'name', 'data-testid', 'aria-label']:
                attr_value = attrs.get(attr_name)
                if attr_value:
                    mapping['by_attributes'][f"{attr_name}:{attr_value}"] = element_data
            
            # 3. Map by text content (for buttons, labels, etc.)
            if text and len(text) > 1:
                # Normalize text for mapping
                import re
                normalized_text = re.sub(r'\s+', ' ', text.strip())
                if normalized_text:
                    mapping['by_text'][normalized_text] = element_data
            
            # 4. Map by input type
            input_type = attrs.get('type', '').lower()
            if input_type:
                type_key = f"{tag}:{input_type}"
                if type_key not in mapping['by_type']:
                    mapping['by_type'][type_key] = []
                mapping['by_type'][type_key].append(element_data)
            
        except Exception as e:
            self.logger.error(f"Failed to add element to stable mapping: {e}")

    def _determine_form_purpose(self, element: Dict, text: str, attrs: Dict) -> Optional[str]:
        """Determine the semantic purpose of a form element"""
        try:
            tag = element.get('tag', '').lower()
            input_type = attrs.get('type', '').lower()
            name = attrs.get('name', '').lower()
            placeholder = attrs.get('placeholder', '').lower()
            aria_label = attrs.get('aria-label', '').lower()
            
            # Combine all text sources for analysis
            all_text = ' '.join([text, name, placeholder, aria_label]).lower()
            
            # Email field detection
            if ('email' in all_text or 
                input_type == 'email' or
                '@' in placeholder):
                return 'email'
            
            # Name field detection
            if ('name' in all_text and 'master' in all_text):
                return 'master_name'
            elif 'name' in all_text:
                return 'name'
            
            # Date field detection  
            if ('date' in all_text and 'birth' in all_text) or input_type == 'date':
                return 'date_of_birth'
            
            # Course field detection
            if 'course' in all_text and tag == 'input':
                return 'course'
            
            # Marriage status (radio buttons)
            if ('married' in all_text or 'marriage' in all_text) and input_type == 'radio':
                if 'yes' in all_text:
                    return 'married_yes'
                elif 'no' in all_text:
                    return 'married_no'
                elif 'maybe' in all_text:
                    return 'married_maybe'
            
            # Course selection (dropdown)
            if (tag == 'select' or 'option' in tag) and 'course' in all_text:
                return 'course_selection'
            
            # Submit button
            if (tag == 'button' or input_type == 'submit') and ('submit' in all_text):
                return 'submit'
            
            # Generic button with specific text
            if tag == 'button' or input_type == 'button':
                if 'submit' in text:
                    return 'submit'
                elif 'next' in text:
                    return 'next'
                elif 'continue' in text:
                    return 'continue'
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to determine form purpose: {e}")
            return None

    async def _merge_with_previous_mapping(self, new_mapping) -> Dict[str, Any]:
        """Merge new mapping with previous stable mapping to maintain consistency"""
        try:
            previous_mapping = self.element_stability['stable_element_map']
            
            # For form purpose mapping, prefer elements that haven't changed indices
            merged_purpose_map = {}
            
            for purpose, new_element in new_mapping.get('by_form_purpose', {}).items():
                if purpose in previous_mapping.get('by_form_purpose', {}):
                    prev_element = previous_mapping['by_form_purpose'][purpose]
                    new_index = new_element.get('index')
                    prev_index = prev_element.get('index')
                    
                    # If indices match, element is stable - keep it
                    if new_index == prev_index:
                        merged_purpose_map[purpose] = new_element
                        self.logger.debug(f"🔒 Stable element {purpose} maintained at index {new_index}")
                    else:
                        # Index changed - use new element but log the change
                        merged_purpose_map[purpose] = new_element
                        self.logger.warning(f"⚠️  Element {purpose} index changed: {prev_index} → {new_index}")
                else:
                    # New element - add it
                    merged_purpose_map[purpose] = new_element
                    self.logger.info(f"➕ New element {purpose} detected at index {new_element.get('index')}")
            
            # Update the mapping
            new_mapping['by_form_purpose'] = merged_purpose_map
            
            return new_mapping
            
        except Exception as e:
            self.logger.error(f"Failed to merge with previous mapping: {e}")
            return new_mapping

    async def _find_element_by_stable_identifier(self, purpose: str, current_state: Dict) -> Optional[int]:
        """Find element using stable identifiers instead of dynamic indices"""
        try:
            stable_elements = current_state.get('stable_elements', {})
            
            # 1. Try form purpose mapping first (most reliable)
            if purpose in stable_elements.get('by_form_purpose', {}):
                element = stable_elements['by_form_purpose'][purpose]
                index = element.get('index')
                self.logger.info(f"🔍 Found {purpose} via purpose mapping → index {index}")
                return index
            
            # 2. Try attribute-based mapping (second most reliable)
            for attr_key, element in stable_elements.get('by_attributes', {}).items():
                if purpose.lower() in attr_key.lower():
                    index = element.get('index')
                    self.logger.info(f"🔍 Found {purpose} via attribute {attr_key} → index {index}")
                    return index
            
            # 3. Try text-based mapping with enhanced lookup
            text_mappings = {
                'married_yes': ['yes', 'married: yes'],
                'married_no': ['no', 'married: no'], 
                'married_maybe': ['maybe', 'married: maybe'],
                'submit': ['submit', 'submit form'],
                'course_selection': ['eag', 'era', 'epai', 'choose'],
                'email': ['email', '@'],
                'name': ['name'],
                'course': ['course'],
                'date_of_birth': ['date', 'birth']
            }
            
            if purpose in text_mappings:
                for text_option in text_mappings[purpose]:
                    # Check exact match first
                    if text_option in stable_elements.get('by_text', {}):
                        element = stable_elements['by_text'][text_option]
                        index = element.get('index')
                        self.logger.info(f"🔍 Found {purpose} via text '{text_option}' → index {index}")
                        return index
                    
                    # Check partial match in text keys
                    for text_key, element in stable_elements.get('by_text', {}).items():
                        if text_option.lower() in text_key.lower():
                            index = element.get('index')
                            self.logger.info(f"🔍 Found {purpose} via partial text match '{text_key}' → index {index}")
                            return index
            
            # 4. Try type-based mapping as fallback
            type_mappings = {
                'email': ['input:email', 'input:text'],
                'date_of_birth': ['input:date', 'input:text'],
                'course': ['input:text'],
                'name': ['input:text'],
                'submit': ['button:submit', 'input:submit']
            }
            
            if purpose in type_mappings:
                for type_key in type_mappings[purpose]:
                    if type_key in stable_elements.get('by_type', {}):
                        elements = stable_elements['by_type'][type_key]
                        if elements and len(elements) > 0:
                            # Take the first matching element
                            element = elements[0]
                            index = element.get('index')
                            self.logger.info(f"🔍 Found {purpose} via type fallback {type_key} → index {index}")
                            return index
            
            #self.logger.warning(f"⚠️  Could not find stable identifier for {purpose}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find stable element for {purpose}: {e}")
            return None

    async def _decide_next_action_intelligent(self, query: str, current_state: Dict, previous_results: List[Dict]) -> Optional[Dict[str, Any]]:
        """Use LLM to intelligently decide the next action based on current state"""
        try:
            from agent.model_manager import ModelManager
            model = ModelManager()
            
            # Extract form data from query if this is a form filling task
            form_data = self._extract_form_data_from_query(query)
            if form_data:
                self.form_state['current_form_data'] = form_data
            
            # Check what form fields still need to be filled
            needed_actions = await self._determine_needed_form_actions(current_state, previous_results)
            
            # If we have specific needed actions, prioritize them
            if needed_actions:
                self.logger.info(f"🎯 Prioritizing needed form actions: {needed_actions}")
                # Use stable element identification for the next needed action
                next_action = await self._create_stable_action(needed_actions[0], current_state)
                if next_action:
                    return next_action
            
            # Safely serialize the current state and previous results
            safe_current_state = self._safe_serialize(current_state)
            safe_previous_results = self._safe_serialize([{
                "step": r.get("step"), 
                "action": r.get("action"), 
                "success": r.get("success"), 
                "reasoning": r.get("reasoning", ""),
                "message": r.get("message", "")
            } for r in previous_results[-5:]])
            
            # Add form state context to the prompt
            form_state_info = {
                'filled_fields': list(self.form_state['filled_fields']),
                'submission_attempted': self.form_state['submission_attempted'],
                'form_data_available': bool(self.form_state['current_form_data'])
            }
            
            prompt = f"""You are an intelligent browser automation agent. Analyze the current state and decide the next action.

ORIGINAL GOAL: {query}

FORM STATE TRACKING:
{json.dumps(form_state_info, indent=2)}

CURRENT PAGE STATE:
{json.dumps(safe_current_state, indent=2)}

PREVIOUS ACTIONS TAKEN:
{json.dumps(safe_previous_results, indent=2)}

IMPORTANT INSTRUCTIONS:
1. Do NOT repeat actions that have already been successful
2. Use stable element identification when possible  
3. Focus on progressing through the form systematically
4. If all fields are filled, focus on submission
5. Avoid infinite loops by checking previous results

Based on the goal and current state, what should be the next action? Consider:
1. Is the goal already achieved? If so, return {{"action": "goal_achieved", "reasoning": "Goal completed because..."}}
2. What specific element or action is needed to progress toward the goal?
3. If there are multiple options, pick the most direct path to the goal.
4. Have we already successfully performed this action before?

Return ONLY a JSON object with this structure:
{{
    "action": "action_name",
    "parameters": {{"param1": "value1"}},
    "reasoning": "Why this action was chosen and how it progresses toward the goal"
}}

Available actions: open_tab, go_to_url, click_element_by_index, input_text, send_keys, scroll_down, scroll_up, get_session_snapshot, get_interactive_elements, get_comprehensive_markdown

IMPORTANT PARAMETER FORMATS:
- For input_text: {{"index": element_index_number, "text": "text_to_type"}}
- For click_element_by_index: {{"index": element_index_number}}
- For go_to_url: {{"url": "https://example.com"}}

For input_text, ALWAYS provide both "index" (numeric element index from interactive elements) and "text" (the text to type).
"""

            response = await model.generate_text(prompt=prompt)
            
            # Parse LLM response
            try:
                response = response.strip().strip("`")
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                    
                action = json.loads(response)
                self.logger.info(f"🧠 LLM decided next action: {action.get('action')} - {action.get('reasoning', '')}")
                
                # Track this action attempt
                self._track_action_attempt(action)
                
                return action
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse LLM action decision: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get intelligent action decision: {e}")
            return None

    def _extract_form_data_from_query(self, query: str) -> Dict[str, str]:
        """Extract form data from the user query"""
        import re
        
        form_data = {}
        
        # Extract common patterns
        email_match = re.search(r"email[^'\"]*['\"]([^'\"]+@[^'\"]+)['\"]", query, re.IGNORECASE)
        if email_match:
            form_data['email'] = email_match.group(1)
        
        name_match = re.search(r"name[^'\"]*['\"]([^'\"]+)['\"]", query, re.IGNORECASE)
        if name_match:
            form_data['name'] = name_match.group(1)
        
        # Extract date of birth
        dob_match = re.search(r"date.{0,20}birth[^'\"]*['\"]([^'\"]+)['\"]", query, re.IGNORECASE)
        if dob_match:
            form_data['date_of_birth'] = dob_match.group(1)
        
        # Extract marital status
        if 'married' in query.lower():
            form_data['married'] = 'yes'
        
        # Extract course information  
        if 'school of ai' in query.lower():
            form_data['course'] = 'The school of AI'
        if 'eag' in query.lower():
            form_data['course_type'] = 'EAG'
        
        return form_data

    async def _determine_needed_form_actions(self, current_state: Dict, previous_results: List[Dict]) -> List[Dict[str, Any]]:
        """Determine what form actions are still needed based on current state and history"""
        needed_actions = []
        
        # Check what's been successfully completed
        successful_actions = [r for r in previous_results if r.get('success', False)]
        
        form_data = self.form_state.get('current_form_data', {})
        
        # Define required form actions based on available data
        required_actions = []
        
        if form_data.get('married') == 'yes':
            required_actions.append({
                'purpose': 'married_yes',
                'action': 'click_element_by_index',
                'description': 'Select "Yes" for married status'
            })
        
        if form_data.get('course'):
            required_actions.append({
                'purpose': 'course',
                'action': 'input_text',
                'text': form_data['course'],
                'description': f'Fill course field with "{form_data["course"]}"'
            })
        
        if form_data.get('course_type'):
            required_actions.append({
                'purpose': 'course_selection',
                'action': 'click_element_by_index',
                'description': f'Select course type "{form_data["course_type"]}"'
            })
        
        if form_data.get('email'):
            required_actions.append({
                'purpose': 'email',
                'action': 'input_text',
                'text': form_data['email'],
                'description': f'Fill email field with "{form_data["email"]}"'
            })
        
        if form_data.get('name'):
            required_actions.append({
                'purpose': 'name',
                'action': 'input_text',
                'text': form_data['name'],
                'description': f'Fill name field with "{form_data["name"]}"'
            })
        
        if form_data.get('date_of_birth'):
            required_actions.append({
                'purpose': 'date_of_birth',
                'action': 'input_text',
                'text': form_data['date_of_birth'],
                'description': f'Fill date of birth with "{form_data["date_of_birth"]}"'
            })
        
        # Check which actions haven't been completed yet
        for action in required_actions:
            purpose = action['purpose']
            
            # Check if this field has been successfully filled
            if purpose not in self.form_state['filled_fields']:
                # Check if we can find the element for this action
                element_index = await self._find_element_by_stable_identifier(purpose, current_state)
                if element_index is not None:
                    action['index'] = element_index
                    needed_actions.append(action)
        
        # Add submit action if all fields are filled and not yet submitted
        if (len(needed_actions) == 0 and 
            len(self.form_state['filled_fields']) > 0 and 
            not self.form_state['submission_attempted']):
            
            submit_index = await self._find_element_by_stable_identifier('submit', current_state)
            if submit_index is not None:
                needed_actions.append({
                    'purpose': 'submit',
                    'action': 'click_element_by_index',
                    'index': submit_index,
                    'description': 'Submit the form'
                })
        
        return needed_actions

    async def _create_stable_action(self, needed_action: Dict, current_state: Dict) -> Optional[Dict[str, Any]]:
        """Create an action using stable element identification"""
        try:
            action_type = needed_action.get('action')
            purpose = needed_action.get('purpose')
            index = needed_action.get('index')
            
            if action_type == 'click_element_by_index' and index is not None:
                return {
                    "action": "click_element_by_index",
                    "parameters": {"index": index},
                    "reasoning": f"Using stable identification to {needed_action.get('description', 'perform action')}"
                }
            
            elif action_type == 'input_text' and index is not None:
                text = needed_action.get('text', '')
                return {
                    "action": "input_text", 
                    "parameters": {"index": index, "text": text},
                    "reasoning": f"Using stable identification to {needed_action.get('description', 'input text')}"
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create stable action: {e}")
            return None

    def _track_action_attempt(self, action: Dict):
        """Track action attempts to prevent infinite loops"""
        action_signature = f"{action.get('action')}:{action.get('parameters', {})}"
        self.form_state['attempted_actions'].append({
            'signature': action_signature,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'action': action
        })
        
        # Keep only recent attempts (last 50)
        if len(self.form_state['attempted_actions']) > 50:
            self.form_state['attempted_actions'] = self.form_state['attempted_actions'][-50:]

    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize objects to JSON-compatible format"""
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, dict):
                return {str(k): self._safe_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._safe_serialize(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return self._safe_serialize(obj.__dict__)
            else:
                return str(obj)
        except Exception:
            return str(obj)

    async def _decide_recovery_action(self, query: str, failed_action: Dict, current_state: Dict) -> Optional[Dict[str, Any]]:
        """Use LLM to decide recovery action when an action fails"""
        try:
            from agent.model_manager import ModelManager
            model = ModelManager()
            
            # Safely serialize the data
            safe_failed_action = self._safe_serialize(failed_action)
            safe_current_state = self._safe_serialize(current_state)
            
            prompt = f"""A browser action failed. Decide on a recovery strategy.

ORIGINAL GOAL: {query}

FAILED ACTION:
{json.dumps(safe_failed_action, indent=2)}

CURRENT PAGE STATE:
{json.dumps(safe_current_state, indent=2)}

What recovery action should be taken? Consider:
1. Is there an alternative way to achieve the same result?
2. Do we need to get more information about the page?
3. Should we try a different approach?

Return ONLY a JSON object:
{{
    "action": "recovery_action_name",
    "parameters": {{"param1": "value1"}},
    "reasoning": "Why this recovery action was chosen"
}}

Return null if no recovery is possible.
"""

            response = await model.generate_text(prompt=prompt)
            
            try:
                response = response.strip().strip("`")
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                if response.lower() == "null":
                    return None
                    
                action = json.loads(response)
                return action
                
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get recovery action: {e}")
            return None

    async def _is_goal_achieved_intelligent(self, original_query: str, results: List[Dict[str, Any]]) -> bool:
        """Use LLM to intelligently determine if the goal has been achieved"""
        try:
            from agent.model_manager import ModelManager
            model = ModelManager()
            
            # Get current page state for analysis
            current_state = await self._get_current_page_state()
            
            # Safely serialize the data
            safe_current_state = self._safe_serialize(current_state)
            safe_results = self._safe_serialize([{
                "action": r.get("action"), 
                "success": r.get("success"), 
                "reasoning": r.get("reasoning"), 
                "message": r.get("message", "")
            } for r in results[-10:]])
            
            prompt = f"""Analyze whether the original goal has been achieved based on the actions taken and current page state.

ORIGINAL GOAL: {original_query}

ACTIONS TAKEN:
{json.dumps(safe_results, indent=2)}

CURRENT PAGE STATE:
{json.dumps(safe_current_state, indent=2)}

Has the original goal been achieved? Consider:
1. Did we successfully navigate to the required page?
2. Did we find and interact with the required elements?
3. Did we extract the required information?
4. Are we on the correct page showing the expected results?

Return ONLY a JSON object:
{{
    "goal_achieved": true/false,
    "reasoning": "Detailed explanation of why the goal is or isn't achieved",
    "confidence": 0.0-1.0
}}
"""

            response = await model.generate_text(prompt=prompt)
            
            try:
                response = response.strip().strip("`")
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                    
                analysis = json.loads(response)
                goal_achieved = analysis.get("goal_achieved", False)
                reasoning = analysis.get("reasoning", "")
                confidence = analysis.get("confidence", 0.0)
                
                self.logger.info(f"🎯 Goal analysis: {goal_achieved} (confidence: {confidence}) - {reasoning}")
                return goal_achieved and confidence > 0.7
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse goal analysis: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to analyze goal achievement: {e}")
            return False

    async def _generate_intelligent_continuation(self, original_query: str, results: List[Dict[str, Any]]) -> Optional[str]:
        """Use LLM to generate continuation strategy if goal not achieved"""
        try:
            from agent.model_manager import ModelManager
            model = ModelManager()
            
            current_state = await self._get_current_page_state()
            
            # Safely serialize the data
            safe_current_state = self._safe_serialize(current_state)
            safe_results = self._safe_serialize([{
                "action": r.get("action"), 
                "success": r.get("success"), 
                "reasoning": r.get("reasoning")
            } for r in results[-10:]])
            
            prompt = f"""The original goal hasn't been fully achieved. Generate a continuation strategy.

ORIGINAL GOAL: {original_query}

ACTIONS TAKEN SO FAR:
{json.dumps(safe_results, indent=2)}

CURRENT PAGE STATE:
{json.dumps(safe_current_state, indent=2)}

What should be the continuation strategy? Consider:
1. What part of the goal is still unfinished?
2. What specific actions are needed to complete it?
3. Are we on the right track or need a different approach?

Return ONLY a JSON object:
{{
    "continuation_needed": true/false,
    "strategy": "Specific description of what needs to be done next",
    "reasoning": "Why this continuation strategy was chosen"
}}
"""

            response = await model.generate_text(prompt=prompt)
            
            try:
                response = response.strip().strip("`")
                if response.lower().startswith("json"):
                    response = response[4:].strip()
                    
                analysis = json.loads(response)
                if analysis.get("continuation_needed", False):
                    return analysis.get("strategy", "")
                return None
                
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate continuation strategy: {e}")
            return None 
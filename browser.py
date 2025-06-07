"""
Browser Control Module - Implementing browser automation using Playwright
"""

import asyncio
import logging
from playwright.async_api import async_playwright
import os
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('browser_controller.log')
    ]
)
logger = logging.getLogger('browser_controller')

class BrowserController:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_initialized = False
    
    async def initialize(self, browser_type='msedge', headless=False, force_new=False):
        """Initialize browser"""
        # If forced to create a new instance, or browser status is abnormal, close existing instances
        if force_new and self.is_initialized:
            logger.info("Forcing creation of new browser instance, closing existing instance")
            await self.close()
        
        # Try to validate existing browser status
        if self.is_initialized:
            try:
                # Try to perform a simple operation to verify the browser is actually available
                logger.info("Checking if browser instance is available...")
                current_url = self.page.url  # This will throw an exception if the browser is unavailable
                logger.info(f"Browser instance status normal, current URL: {current_url}")
                return {"status": "success", "message": "Browser is already initialized and status is normal"}
            except Exception as e:
                logger.warning(f"Browser status abnormal, will reinitialize: {str(e)}")
                # Browser status abnormal, reset status and reinitialize
                await self.close()
        
        try:
            logger.info(f"Starting to initialize {browser_type} browser, headless={headless}")
            self.playwright = await async_playwright().start()
            
            # Add pre-browser launch log
            logger.info("Launching browser...")
            
            # Configure browser launch parameters for improved stability
            browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
            
            # Launch different browsers based on browser type
            if browser_type.lower() == 'chromium':
                browser_instance = self.playwright.chromium
                self.browser = await browser_instance.launch(
                    headless=headless,
                    args=browser_args
                )
            elif browser_type.lower() == 'edge' or browser_type.lower() == 'msedge':
                browser_instance = self.playwright.chromium
                self.browser = await browser_instance.launch(
                    headless=headless,
                    channel="msedge",
                    args=browser_args
                )
            elif browser_type.lower() == 'firefox':
                browser_instance = self.playwright.firefox
                self.browser = await browser_instance.launch(headless=headless)
            elif browser_type.lower() == 'webkit':
                browser_instance = self.playwright.webkit
                self.browser = await browser_instance.launch(headless=headless)
            else:
                logger.error(f"Unsupported browser type: {browser_type}")
                return {"status": "error", "message": f"Unsupported browser type: {browser_type}"}
            
            # Add post-browser launch log
            logger.info(f"{browser_type} browser launch completed")
            
            # Set up browser context
            logger.info("Creating browser context...")
            self.context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 800}
            )
            
            # Create new page
            logger.info("Creating new page...")
            self.page = await self.context.new_page()
            
            # Set page event listeners for debugging
            self.page.on("console", lambda msg: logger.info(f"Browser console: {msg.text}"))
            self.page.on("pageerror", lambda err: logger.error(f"Page error: {err}"))
            
            # Verify page is accessible
            await self.page.goto("about:blank")
            logger.info("Page created successfully and is accessible")
            
            self.is_initialized = True
            
            # Add a visible operation to confirm the browser window is actually open
            if not headless:
                # Open an obvious page to confirm visibility
                await self.page.goto("data:text/html,<html><body><h1 style='color:red; font-size:50px;'>Browser Control System Started</h1></body></html>")
                logger.info("Browser displaying test page")
                await asyncio.sleep(1)  # Brief pause to let the user see
            
            return {"status": "success", "message": f"Browser initialization successful: {browser_type}"}
        
        except Exception as e:
            logger.error(f"Browser initialization failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Ensure all resources are released
            try:
                if self.page:
                    await self.page.close()
                if self.context:
                    await self.context.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()
            except Exception as close_err:
                logger.error(f"Error cleaning up resources: {str(close_err)}")
            
            # Reset status
            self.playwright = None
            self.browser = None
            self.context = None
            self.page = None
            self.is_initialized = False
            
            return {"status": "error", "message": f"Browser initialization failed: {str(e)}"}
    
    async def navigate(self, url):
        """Navigate to specified URL, reinitializing browser if necessary"""
        # Check browser health status
        health_result = await self.check_browser_health(auto_reinitialize=True)
        if health_result["status"] == "error":
            return health_result
        
        try:
            logger.info(f"Navigating to: {url}")
            await self.page.goto(url)
            current_url = self.page.url
            logger.info(f"Navigation complete, current URL: {current_url}")
            return {"status": "success", "message": f"Navigated to: {current_url}"}
        
        except Exception as e:
            logger.error(f"Navigation to {url} failed: {str(e)}")
            return {"status": "error", "message": f"Navigation to {url} failed: {str(e)}"}

    async def login_cvat(self, username="admin", password="Yyh277132984"):
        """Login to CVAT - Adapting to two-step login flow, using Chrome browser"""
        # First check browser health status
        health_result = await self.check_browser_health(auto_reinitialize=True, browser_type="msedge")
        if health_result["status"] == "error":
            return health_result
        
        try:
            # Ensure navigation to login page
            await self.navigate("http://localhost:8080/auth/login")
            
            # Wait for page to fully load
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            # Step 1: Clear and enter username
            logger.info("Step 1: Enter username")
            username_field = await self.page.query_selector('input[placeholder="Email or username"]')
            if not username_field:
                username_field = await self.page.query_selector('input[type="text"]')
            
            if username_field:
                # Clear field first
                await username_field.click()
                await username_field.press('Control+a')
                await username_field.press('Delete')
                
                # Enter username
                await username_field.type(username)
                logger.info(f"Username entered: {username}")
                
                # Press Enter to submit username
                await username_field.press('Enter')
                logger.info("Pressed Enter to submit username")
                
                # Wait for password input field to appear
                await asyncio.sleep(2)
                
                # Step 2: Enter password
                logger.info("Step 2: Find and enter password field")
                password_field = await self.page.query_selector('input[type="password"]')
                
                if password_field:
                    logger.info("Password input field found")
                    # Ensure password field has focus
                    await password_field.click()
                    
                    # Clear any existing content
                    await password_field.press('Control+a')
                    await password_field.press('Delete')
                    
                    # Enter password
                    await password_field.type(password)
                    logger.info("Password entered")
                    
                    # Click "Next" button
                    next_button = await self.page.query_selector('button:has-text("Next")')
                    if next_button:
                        await next_button.click()
                        logger.info("Clicked Next button")
                    else:
                        await password_field.press('Enter')
                    
                    # Wait for login to complete
                    await asyncio.sleep(5)
                    
                    # Check if login was successful
                    is_logged_in = await self.check_login_success()
                    if is_logged_in:
                        return {"status": "success", "message": "Successfully logged in to CVAT"}
                    else:
                        return {"status": "warning", "message": "Login operation executed, but may not have been successful, please check browser"}
                else:
                    logger.error("Password input field not found")
                    return {"status": "error", "message": "Password input field not found"}
            else:
                logger.error("Username input field not found")
                return {"status": "error", "message": "Username input field not found"}
        
        except Exception as e:
            logger.error(f"Error during CVAT login: {str(e)}")
            return {"status": "error", "message": f"CVAT login failed: {str(e)}"}

    async def check_login_success(self):
        """Check if successfully logged in to CVAT"""
        try:
            logged_in_element = await self.page.query_selector('.cvat-header-menu-user-dropdown')
            return logged_in_element is not None
        except Exception:
            return False

    async def create_project(self, name="New Project", labels=None):
        """Create new project"""
        if not self.is_initialized:
            init_result = await self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        if labels is None:
            labels = [{"name": "person"}, {"name": "car"}]
        
        try:
            # Navigate to projects page
            await self.navigate("http://localhost:8080/projects")
            
            # Click create project button
            await self.page.click('.cvat-create-project-button')
            
            # Fill in project name
            await self.page.fill('input#name', name)
            
            # Add labels
            for label in labels:
                await self.page.click('.cvat-constructor-viewer-new-item')
                await self.page.fill('.cvat-label-constructor-creator input', label["name"])
                await self.page.click('.cvat-label-constructor-creator-submit-button')
            
            # Submit creation
            await self.page.click('.cvat-submit-project-button')
            
            # Wait for creation to succeed
            await self.page.wait_for_selector('.cvat-project-page', timeout=10000)
            logger.info(f"Successfully created project: {name}")
            return {"status": "success", "message": f"Successfully created project: {name}"}
        
        except Exception as e:
            logger.error(f"Failed to create project: {str(e)}")
            return {"status": "error", "message": f"Failed to create project: {str(e)}"}

    async def open_annotation_interface(self, task_id, job_id=1):
        """Open CVAT annotation interface"""
        try:
            # Force close old session to ensure fresh state
            logger.info("Initializing new browser session...")
            await self.close()  # Close any potentially existing old browser instances
            init_result = await self.initialize(browser_type="msedge", headless=False, force_new=True)
            if init_result["status"] == "error":
                return init_result

            # Login to CVAT
            logger.info("Logging in to CVAT...")
            await self.navigate("http://localhost:8080/auth/login")
            login_result = await self.login_cvat()
            if login_result["status"] == "error":
                return login_result

            # After successful login, navigate to annotation interface (task and job used consecutively)
            url = f"http://localhost:8080/tasks/{task_id}/jobs/{job_id}"
            logger.info(f"Navigating to annotation interface: {url}")
            await self.navigate(url)

            # Wait for annotation interface to load
            logger.info("Waiting for annotation interface to load...")
            await self.page.wait_for_selector('.cvat-canvas-container', timeout=20000)
            
            return {
                "status": "success",
                "message": f"Opened annotation interface for task {task_id} (Job #{job_id})"
            }
        except Exception as e:
            logger.error(f"Failed to open annotation interface: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"Failed to open annotation interface: {str(e)}"}

    async def check_browser_health(self, auto_reinitialize=True, browser_type='msedge'):
        """Check browser health status, reinitialize automatically if necessary"""
        if not self.is_initialized:
            logger.info("Browser not initialized")
            if auto_reinitialize:
                logger.info("Automatically reinitializing browser")
                return await self.initialize(browser_type=browser_type)
            return {"status": "error", "message": "Browser not initialized"}
        
        try:
            # Try to perform simple operations to verify browser is available
            logger.info("Checking browser instance health status...")
            
            # Check if browser process still exists
            if not self.browser or not self.page:
                raise Exception("Browser or page object doesn't exist")
            
            # Try to perform simple operation
            is_connected = self.browser.is_connected()  # Removed await
            if not is_connected:
                raise Exception("Browser connection has been lost")
            
            # Try to access current URL (this may throw exception if page is closed)
            try:
                current_url = self.page.url
                logger.info(f"Browser status normal, current URL: {current_url}")
            except Exception as e:
                raise Exception(f"Unable to get page URL: {str(e)}")
                
            return {"status": "success", "message": "Browser health status normal"}
        
        except Exception as e:
            logger.warning(f"Browser status abnormal: {str(e)}")
            
            # Reset browser status
            try:
                await self.close()
            except Exception as close_err:
                logger.error(f"Error closing abnormal browser: {str(close_err)}")
            
            # If auto-reinitialization is needed
            if auto_reinitialize:
                logger.info("Automatically reinitializing browser")
                return await self.initialize(browser_type=browser_type)
            
            return {"status": "error", "message": f"Browser status abnormal: {str(e)}"}    


    async def close(self):
        if not self.is_initialized:
            logger.info("No browser to close")
            return {"status": "success", "message": "No browser to close"}

        # First check if browser is already disconnected
        if self.browser is not None:
            try:
                if not self.browser.is_connected():
                    logger.info("Detected browser already disconnected, directly resetting state")
                    self.page = None
                    self.context = None
                    self.browser = None
                    self.playwright = None
                    self.is_initialized = False
                    return {"status": "success", "message": "Browser already disconnected, state reset"}
            except Exception as check_err:
                logger.warning(f"Error detecting browser status: {str(check_err)}")
        
        # Define internal function close_all here to close all components
        async def close_all():
            if self.page:
                try:
                    await self.page.close()
                    logger.info("Page closed")
                except Exception as e:
                    logger.warning(f"Error closing page: {str(e)}")
            if self.context:
                try:
                    await self.context.close()
                    logger.info("Context closed")
                except Exception as e:
                    logger.warning(f"Error closing context: {str(e)}")
            if self.browser:
                try:
                    await self.browser.close()
                    logger.info("Browser closed")
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
            if self.playwright:
                try:
                    await self.playwright.stop()
                    logger.info("Playwright stopped")
                except Exception as e:
                    logger.warning(f"Error stopping Playwright: {str(e)}")
        
        # Use asyncio.wait_for to set a 1 second timeout for closing operations
        try:
            await asyncio.wait_for(close_all(), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Close operation timed out, but will directly reset state")
        
        # Reset all states
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self.is_initialized = False
        logger.info("Browser state reset")
        return {"status": "success", "message": "Browser closed or state reset"}

    async def take_screenshot(self, path="screenshot.png"):
        """Take a screenshot"""
        if not self.is_initialized:
            init_result = await self.initialize()
            if init_result["status"] == "error":
                return init_result
        
        try:
            await self.page.screenshot(path=path)
            logger.info(f"Screenshot saved to: {path}")
            return {"status": "success", "message": f"Screenshot saved to: {path}"}
        
        except Exception as e:
            logger.error(f"Failed to take screenshot: {str(e)}")
            return {"status": "error", "message": f"Failed to take screenshot: {str(e)}"}
        
# Create a global browser controller instance
browser_controller = BrowserController()

# Async function to handle browser requests
async def handle_browser_request(request_data):
    """Handle browser control requests from the client"""
    try:
        action = request_data.get("action")
        
        if action == "open_cvat":
            logger.info("Starting to execute open_cvat operation...")
            # Force close existing browser instance regardless of previous state
            if browser_controller.is_initialized:
                logger.info("Detected existing browser instance, performing close operation")
                await browser_controller.close()
            
            # Initialize new browser: pass relevant parameters, ensure non-headless mode for debugging
            browser_type = request_data.get("browser_type", "msedge")
            init_result = await browser_controller.initialize(browser_type=browser_type, headless=False, force_new=True)
            logger.info(f"Browser initialization result: {init_result}")
            if init_result.get("status") == "error":
                return init_result
            
            # Navigate to CVAT login page
            nav_result = await browser_controller.navigate("http://localhost:8080/auth/login")
            logger.info(f"Navigation result: {nav_result}")
            if nav_result.get("status") == "error":
                return nav_result
            
            # Try to login to CVAT
            try:
                login_result = await browser_controller.login_cvat()
                logger.info(f"CVAT login return result: {login_result}")
                return login_result
            except Exception as e:
                logger.warning(f"Opened CVAT but login may have failed: {str(e)}")
                return {"status": "partial", "message": "CVAT opened but login may have failed, please login manually"}
        
        # Other branches remain unchanged...
        elif action == "navigate":
            url = request_data.get("url")
            if not url:
                logger.error("Navigation operation requires URL parameter")
                return {"status": "error", "message": "Navigation operation requires URL parameter"}
            return await browser_controller.navigate(url)
        elif action == "login_cvat":
            username = request_data.get("username", "admin")
            password = request_data.get("password", "Yyh277132984")
            return await browser_controller.login_cvat(username, password)
        elif action == "create_project":
            name = request_data.get("name", "New Project")
            labels = request_data.get("labels", None)
            return await browser_controller.create_project(name, labels)
        elif action == "open_annotation":
            task_id = request_data.get("task_id")
            job_id = request_data.get("job_id", 1)  # Default is 1
            
            if not task_id:
                logger.error("Open annotation interface operation requires task ID")
                return {"status": "error", "message": "Open annotation interface operation requires task ID"}
            
            return await browser_controller.open_annotation_interface(task_id, job_id)
            
            return await browser_controller.open_annotation_interface(id_value, type_value, job_id)
        elif action == "close":
            return await browser_controller.close()
        elif action == "take_screenshot":
            path = request_data.get("path", "screenshot.png")
            return await browser_controller.take_screenshot(path)
        else:
            logger.error(f"Unknown operation: {action}")
            return {"status": "error", "message": f"Unknown operation: {action}"}
    
    except Exception as e:
        logger.error(f"Error handling browser request: {str(e)}")
        return {"status": "error", "message": f"Error handling browser request: {str(e)}"}
    

# Synchronous wrapper function for use in non-async environments
def process_browser_request(request_data):
    """
    Synchronous wrapper function for processing browser requests.
    Uses asyncio.run to ensure a new event loop is created in the main thread,
    avoiding problems caused by existing event loops.
    """
    try:
        logger.info("Using asyncio.run to execute asynchronous browser request")
        result = asyncio.run(handle_browser_request(request_data))
        logger.info(f"Async request returned result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing browser request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": f"Error processing browser request: {str(e)}"}


# Simplified function: Open CVAT
def open_cvat():
    """Simplified function to open CVAT"""
    request_data = {"action": "open_cvat"}
    return process_browser_request(request_data)

# Simplified function: Open annotation interface
def open_annotation_interface(task_id, job_id=1):
    """Simplified function to open CVAT annotation interface"""
    request_data = {"action": "open_annotation", "task_id": task_id, "job_id": job_id}
    return process_browser_request(request_data)


# Test code
if __name__ == "__main__":
    # Test opening CVAT in browser
    result = open_cvat()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # Wait for user to view
    input("Press Enter to close browser...")
    
    # Close browser
    close_result = process_browser_request({"action": "close"})
    print(json.dumps(close_result, indent=2, ensure_ascii=False))
    
// Helper to initialize the ChatKit Web Component
//
// This script waits for the DOM to load, then attaches a click handler to the
// “Start Chat” button.  When clicked it requests a ChatKit session from the
// backend and mounts the <openai-chatkit> element into the page.  The
// component accepts attributes such as `client-token` and `workflow-id` to
// configure the chat.  For more options see the official documentation.

document.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('startChat');
  const container = document.getElementById('chatContainer');
  button.addEventListener('click', async () => {
    button.disabled = true;
    button.textContent = 'Loading...';
    try {
      const resp = await fetch('/api/chatkit/session');
      if (!resp.ok) {
        throw new Error('Failed to fetch ChatKit session');
      }
      const { client_secret } = await resp.json();
      // Create the ChatKit component
      const chatEl = document.createElement('openai-chatkit');
      chatEl.setAttribute('client-token', client_secret);
      // Optional: specify your workflow ID here if needed
      // chatEl.setAttribute('workflow-id', 'YOUR_WORKFLOW_ID');
      chatEl.style.width = '100%';
      chatEl.style.height = '600px';
      // Clear any existing content and append the chat
      container.innerHTML = '';
      container.appendChild(chatEl);
    } catch (err) {
      console.error(err);
      alert('Error initializing chat: ' + err.message);
      button.disabled = false;
      button.textContent = 'Start Chat';
    }
  });
});
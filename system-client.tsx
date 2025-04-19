import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { MessageSquare, Send, Upload, Download } from 'lucide-react';

interface SystemMessage {
  type: string;
  content: any;
}

interface SystemMetrics {
  memoryStatus: number;
  cpuLoad: number;
  networkStatus: number;
  overallHealth: number;
  processing: boolean;
}

function SystemClient() {
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<SystemMessage[]>([]);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sysMetrics, setSysMetrics] = useState<SystemMetrics>({
    memoryStatus: 0,
    cpuLoad: 0,
    networkStatus: 0,
    overallHealth: 0,
    processing: false
  });
  const [logs, setLogs] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [debug, setDebug] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileUploadRef = useRef<HTMLInputElement>(null);
  const [urlInput, setUrlInput] = useState('');
  const [promptInput, setPromptInput] = useState('');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 5;

  const handleConnectionClose = () => {
    if (wsConnection) {
      wsConnection.close();
    }

    if (reconnectAttempts < maxReconnectAttempts) {
      setReconnectAttempts(prev => prev + 1);
      setTimeout(connectWebSocket, 2000 * reconnectAttempts);
      setConnectionStatus(`Connection lost. Reconnecting (${reconnectAttempts + 1}/${maxReconnectAttempts})...`);
    } else {
      setConnectionStatus('Disconnected');
      setErrorMessage('Maximum reconnection attempts reached. Please refresh the page.');
    }
  };

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => {
      setConnectionStatus('Connected');
      setErrorMessage(null);
      setWsConnection(ws);
      setReconnectAttempts(0);
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        setWsConnection(ws);
        processServerMessage(data);
      } catch (error) {
        setErrorMessage(`Failed to parse server message: ${error instanceof Error ? error.message : String(error)}`);
      }
    };

    ws.onclose = handleConnectionClose;

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setErrorMessage(`WebSocket error: ${error instanceof Error ? error.message : 'Connection failed'}`);
    };
  };

  const processServerMessage = (data: any) => {
    if (!data || typeof data !== 'object') {
      setErrorMessage('Received invalid data from server');
      return;
    }

    if (data.type === 'logs') {
      setLogs(prev => [...prev, data.content]);
    } else if (data.type === 'warnings') {
      setWarnings(prev => [...prev, data.content]);
    } else if (data.type === 'errors') {
      setErrors(prev => [...prev, data.content]);
    } else if (data.type === 'debug') {
      setDebug(prev => [...prev, data.content]);
    } else if (data.type === 'system_metrics') {
      setSysMetrics(data.content);
    } else {
      setMessages(prev => [...prev, { type: data.type || 'unknown', content: data.content }]);
    }

    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  const sendMessage = () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      setErrorMessage('WebSocket is not connected');
      return;
    }

    if (!inputText.trim()) return;

    const message = { type: 'user_input', content: inputText };
    wsConnection.send(JSON.stringify(message));

    setMessages(prev => [...prev, { type: 'user', content: inputText }]);
    setInputText('');
  };

  const handleFileUpload = () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      setErrorMessage('Cannot upload file: WebSocket is not connected');
      return;
    }

    if (!fileUploadRef.current || !fileUploadRef.current.files || fileUploadRef.current.files.length === 0) {
      setErrorMessage('No file selected');
      return;
    }

    const file = fileUploadRef.current.files[0];

    if (file.size > 5 * 1024 * 1024) {
      setErrorMessage('File is too large. Maximum size is 5MB.');
      return;
    }

    const reader = new FileReader();

    reader.onload = (event: ProgressEvent<FileReader>) => {
      if (event.target && event.target.result) {
        const fileContent = event.target.result.toString();
        const message = {
          type: 'file_upload',
          filename: file.name,
          content: fileContent
        };

        wsConnection.send(JSON.stringify(message));

        setMessages(prev => [...prev, {
          type: 'system',
          content: `File uploaded: ${file.name}`
        }]);

        setErrorMessage(`File ${file.name} uploaded successfully.`);
      }
    };

    reader.onerror = (event) => {
      setErrorMessage(`Error reading file: ${event instanceof Error ? event.message : 'Unknown error'}`);
    };

    reader.readAsText(file);
  };

  const handleDocDownload = () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      setErrorMessage('Cannot upload file: WebSocket is not connected');
      return;
    }

    if (!fileUploadRef.current || !fileUploadRef.current.files || fileUploadRef.current.files.length === 0) {
      setErrorMessage('No file selected');
      return;
    }

    const file = fileUploadRef.current.files[0];

    if (file.size > 5 * 1024 * 1024) {
      setErrorMessage('File is too large. Maximum size is 5MB.');
      return;
    }

    const reader = new FileReader();

    reader.onload = (event: ProgressEvent<FileReader>) => {
      if (event.target && event.target.result) {
        const fileContent = event.target.result.toString();
        const message = {
          type: 'document_download',
          filename: file.name,
          content: fileContent
        };

        wsConnection.send(JSON.stringify(message));

        setMessages(prev => [...prev, {
          type: 'system',
          content: `Document processed: ${file.name}`
        }]);

        setErrorMessage(`Document ${file.name} is being processed.`);
      }
    };

    reader.onerror = (event) => {
      setErrorMessage(`Error reading file: ${event instanceof Error ? event.message : 'Unknown error'}`);
    };

    reader.readAsText(file);
  };

  const handleWebCrawl = () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      setErrorMessage('Cannot crawl: WebSocket is not connected');
      return;
    }

    if (!urlInput.trim()) {
      setErrorMessage('Please enter a URL to crawl');
      return;
    }

    if (!urlInput.match(/^https?:\/\/.+\..+/)) {
      setErrorMessage('Please enter a valid URL (e.g., https://example.com)');
      return;
    }

    const message = { type: 'web_crawl', url: urlInput };
    wsConnection.send(JSON.stringify(message));

    setMessages(prev => [...prev, {
      type: 'system',
      content: `Web crawling started: ${urlInput}`
    }]);

    setErrorMessage(`Web crawling started for ${urlInput}`);
    setUrlInput('');
  };

  const handleSendPrompt = () => {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
      setErrorMessage('Cannot send prompt: WebSocket is not connected');
      return;
    }

    if (!promptInput.trim()) {
      setErrorMessage('Please enter a prompt');
      return;
    }

    const message = { type: 'prompt', content: promptInput };
    wsConnection.send(JSON.stringify(message));

    setMessages(prev => [...prev, {
      type: 'user',
      content: promptInput
    }]);

    setErrorMessage(`Prompt sent: ${promptInput}`);
    setPromptInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const renderMessages = () => {
    return messages.map((msg, index) => (
      <div key={index} className={`message ${msg.type === 'user' ? 'user-message' : 'system-message'}`}>
        <div className="message-header">
          {msg.type === 'user' ? 'You' : 'System'}
        </div>
        <div className="message-content">
          {typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}
        </div>
      </div>
    ));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
  };

  const adjustInputHeight = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    target.style.height = 'auto';
    target.style.height = `${target.scrollHeight}px`;
  };

  const autoResizeTextarea = (e: React.FocusEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    const previousSibling = target.previousSibling as HTMLElement;

    if (previousSibling && previousSibling.classList.contains('input-placeholder')) {
      previousSibling.style.display = 'none';
    }

    adjustInputHeight(e as unknown as React.ChangeEvent<HTMLTextAreaElement>);
  };

  const showPlaceholder = (e: React.FocusEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    const previousSibling = target.previousSibling as HTMLElement;

    if (previousSibling && previousSibling.classList.contains('input-placeholder') && !target.value) {
      previousSibling.style.display = 'block';
    }
  };

  const renderMetricsChart = () => {
    const data = [
      { name: 'Memory', value: sysMetrics.memoryStatus },
      { name: 'CPU', value: sysMetrics.cpuLoad },
      { name: 'Network', value: sysMetrics.networkStatus },
      { name: 'Health', value: sysMetrics.overallHealth }
    ];

    return (
      <div className="metrics-chart">
        <h3>System Metrics {sysMetrics.processing && <span className="processing-indicator">Processing</span>}</h3>
        <LineChart width={500} height={300} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
        <div className="metrics-details">
          <div className="metric-item">Memory: {sysMetrics.memoryStatus}%</div>
          <div className="metric-item">CPU: {sysMetrics.cpuLoad}%</div>
          <div className="metric-item">Network: {sysMetrics.networkStatus}%</div>
          <div className="metric-item">Overall Health: {sysMetrics.overallHealth}%</div>
        </div>
      </div>
    );
  };

  return (
    <div className="system-client">
      <header className="client-header">
        <h1>AI System Interface</h1>
        <div className={`connection-status ${connectionStatus === 'Connected' ? 'connected' : 'disconnected'}`}>
          {connectionStatus}
        </div>
      </header>

      <div className="client-body">
        <main className="main-area">
          <div className="chat-container">
            <div className="messages-container">
              {renderMessages()}
              <div ref={messagesEndRef} />
            </div>

            <div className="input-container">
              <div className="input-placeholder">Type a message...</div>
              <textarea
                value={inputText}
                onChange={handleInputChange}
                onKeyDown={handleKeyPress}
                onInput={adjustInputHeight}
                onFocus={autoResizeTextarea}
                onBlur={showPlaceholder}
                rows={1}
              />
              <button onClick={sendMessage} className="send-button">
                <Send size={20} />
              </button>
            </div>
          </div>

          <div className="metrics-container">
            {renderMetricsChart()}
          </div>
        </main>

        <aside className="side-panel">
          <div className="panel-section">
            <h3>File Operations</h3>
            <input
              type="file"
              ref={fileUploadRef}
              style={{ display: 'none' }}
              onChange={() => fileUploadRef.current?.click()}
            />
            <button onClick={() => fileUploadRef.current?.click()} className="action-button">
              <Upload size={16} /> Upload File
            </button>
            <button onClick={handleDocDownload} className="action-button">
              <Download size={16} /> Process Document
            </button>
          </div>

          <div className="panel-section">
            <h3>Web Crawling</h3>
            <input
              type="text"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              placeholder="Enter URL to crawl"
              className="text-input"
            />
            <button onClick={handleWebCrawl} className="action-button">
              Crawl Web
            </button>
          </div>

          <div className="panel-section">
            <h3>AI Prompt</h3>
            <textarea
              value={promptInput}
              onChange={(e) => setPromptInput(e.target.value)}
              placeholder="Enter prompt for AI"
              className="text-input prompt-input"
              rows={4}
            />
            <button onClick={handleSendPrompt} className="action-button">
              Send Prompt
            </button>
          </div>

          <div className="panel-section logs-section">
            <h3>System Logs</h3>
            <div className="logs-container">
              <div className="log-category">
                <h4>Debug ({debug.length})</h4>
                <div className="log-entries">
                  {debug.slice(-5).map((entry, i) => (
                    <div key={i} className="log-entry debug">{entry}</div>
                  ))}
                </div>
              </div>

              <div className="log-category">
                <h4>Logs ({logs.length})</h4>
                <div className="log-entries">
                  {logs.slice(-5).map((entry, i) => (
                    <div key={i} className="log-entry info">{entry}</div>
                  ))}
                </div>
              </div>

              <div className="log-category">
                <h4>Warnings ({warnings.length})</h4>
                <div className="log-entries">
                  {warnings.slice(-5).map((entry, i) => (
                    <div key={i} className="log-entry warning">{entry}</div>
                  ))}
                </div>
              </div>

              <div className="log-category">
                <h4>Errors ({errors.length})</h4>
                <div className="log-entries">
                  {errors.slice(-5).map((entry, i) => (
                    <div key={i} className="log-entry error">{entry}</div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {errorMessage && (
        <div className="error-toast">
          {errorMessage}
          <button onClick={() => setErrorMessage(null)} className="close-button">×</button>
        </div>
      )}
    </div>
  );
}

export default SystemClient;

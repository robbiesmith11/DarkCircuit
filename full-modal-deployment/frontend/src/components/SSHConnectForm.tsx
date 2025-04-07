import React, { useState } from 'react';
import { Terminal as TerminalIcon, Lock, Server, User, Key } from 'lucide-react';

interface SSHConnectProps {
  onConnect: (data: {
    host: string;
    port: number;
    username: string;
    password: string;
    keyPath?: string;
  }) => Promise<void>;
  isConnecting: boolean;
  isConnected: boolean;
  connectionError: string | null;
}

export const SSHConnectForm: React.FC<SSHConnectProps> = ({
  onConnect,
  isConnecting,
  isConnected,
  connectionError
}) => {
  const [host, setHost] = useState('');
  const [port, setPort] = useState('22');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [useKeyAuth, setUseKeyAuth] = useState(false);
  const [keyPath, setKeyPath] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    try {
      await onConnect({
        host,
        port: parseInt(port, 10),
        username,
        password,
        keyPath: useKeyAuth ? keyPath : undefined
      });
    } catch (error) {
      console.error('Error connecting to SSH:', error);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-md border border-cyan">
      <h2 className="text-xl text-white mb-4 flex items-center">
        <Server className="mr-2" size={20} />
        Connect to External Container
      </h2>

      {connectionError && (
        <div className="bg-red-900 text-white p-3 rounded-md mb-4">
          {connectionError}
        </div>
      )}

      {isConnected ? (
        <div className="bg-green-900 text-white p-3 rounded-md">
          Connected to SSH server at {host}
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex items-center space-x-2">
            <div className="flex-1">
              <label className="flex items-center text-gray-300 mb-1">
                <Server size={16} className="mr-1" />
                Host or IP Address
              </label>
              <input
                type="text"
                value={host}
                onChange={(e) => setHost(e.target.value)}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-bgCyan"
                required
              />
            </div>
            <div className="w-24">
              <label className="flex items-center text-gray-300 mb-1">
                Port
              </label>
              <input
                type="number"
                value={port}
                onChange={(e) => setPort(e.target.value)}
                placeholder="22"
                className="w-full bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-bgCyan"
                required
              />
            </div>
          </div>

          <div>
            <label className="flex items-center text-gray-300 mb-1">
              <User size={16} className="mr-1" />
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-bgCyan"
              required
            />
          </div>

          <div className="flex items-center space-x-2 text-gray-300">
            <input
              type="checkbox"
              id="useKeyAuth"
              checked={useKeyAuth}
              onChange={(e) => setUseKeyAuth(e.target.checked)}
              className="bg-gray-700 rounded focus:ring-bgCyan"
            />
            <label htmlFor="useKeyAuth" className="flex items-center">
              <Key size={16} className="mr-1" />
              Use SSH Key Instead of Password
            </label>
          </div>

          {useKeyAuth ? (
            <div>
              <label className="flex items-center text-gray-300 mb-1">
                <Key size={16} className="mr-1" />
                Key Path (on server)
              </label>
              <input
                type="text"
                value={keyPath}
                onChange={(e) => setKeyPath(e.target.value)}
                placeholder="/path/to/private/key"
                className="w-full bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-bgCyan"
                required={useKeyAuth}
              />
            </div>
          ) : (
            <div>
              <label className="flex items-center text-gray-300 mb-1">
                <Lock size={16} className="mr-1" />
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-gray-700 text-white px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-bgCyan"
                required={!useKeyAuth}
              />
            </div>
          )}

          <button
            type="submit"
            disabled={isConnecting}
            className="w-full bg-cyan hover:bg-bgCyan text-black py-2 px-4 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 flex items-center justify-center"
          >
            {isConnecting ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Connecting...
              </>
            ) : (
              <>
                <TerminalIcon className="mr-2" size={18} />
                Connect
              </>
            )}
          </button>
        </form>
      )}
    </div>
  );
};

export default SSHConnectForm;
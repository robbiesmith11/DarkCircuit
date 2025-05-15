import React from 'react';
import {
  Globe2,
  Network,
  Search,
  ShieldCheck,
  Workflow,
  FileCode,
  Info
} from 'lucide-react';

interface ToolShortcutsProps {
  onToolClick: (command: string) => void;
}

export const ToolShortcuts: React.FC<ToolShortcutsProps> = ({ onToolClick }) => {
  const tools = [
    {
      name: 'Network Scan',
      icon: <Network size={16} />,
      command: 'nmap -sV -F 127.0.0.1',
      description: 'Quick service scan on localhost'
    },
    {
      name: 'DNS Lookup',
      icon: <Globe2 size={16} />,
      command: 'dig google.com',
      description: 'DNS lookup for google.com'
    },
    {
      name: 'VPN Status',
      icon: <ShieldCheck size={16} />,
      command: 'ps aux | grep openvpn | grep -v grep && ip a show tun0 && echo "\nRouting:" && ip route show',
      description: 'Check VPN connection status'
    },
    {
      name: 'Public IP',
      icon: <Info size={16} />,
      command: 'curl -s https://ifconfig.me && echo ""',
      description: 'Show your public IP address'
    },
    {
      name: 'Services',
      icon: <Workflow size={16} />,
      command: 'netstat -tuln | grep LISTEN',
      description: 'List listening services'
    },
    {
      name: 'Find SUID',
      icon: <Search size={16} />,
      command: 'echo "This would search for SUID binaries (disabled in sandbox)"',
      description: 'List SUID binaries'
    },
    {
      name: 'Encode Base64',
      icon: <FileCode size={16} />,
      command: 'echo "Hello World" | base64',
      description: 'Base64 encode a string'
    },
    {
      name: 'Decode Base64',
      icon: <FileCode size={16} />,
      command: 'echo "SGVsbG8gV29ybGQK" | base64 -d',
      description: 'Base64 decode a string'
    }
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h2 className="text-white text-lg font-semibold mb-4">Tool Shortcuts</h2>
      <div className="grid grid-cols-4 gap-2">
        {tools.map((tool, index) => (
          <button
            key={index}
            onClick={() => onToolClick(tool.command)}
            className="bg-gray-700 hover:bg-gray-600 text-white p-2 rounded-lg text-xs flex flex-col items-center justify-center h-24"
            title={tool.description}
          >
            <div className="bg-gray-600 p-2 rounded-full mb-2">
              {tool.icon}
            </div>
            <span>{tool.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
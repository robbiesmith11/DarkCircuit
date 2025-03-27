import React from 'react';
import { Server } from 'lucide-react';

interface SimplifiedSidebarProps {
  onServiceToggle?: (service: string) => void;
}

export const Sidebar: React.FC<SimplifiedSidebarProps> = ({
  onServiceToggle = () => {},
}) => {
  return (
    <div className="w-64 bg-gray-800 text-white p-4 flex flex-col h-screen">
      <h1 className="text-2xl font-bold mb-6">AI Cyber Labs</h1>

      <div className="mb-8">
        <h2 className="text-xl font-semibold text-lime-500 mb-4">DarkCircuit</h2>
        <div className="bg-gray-700 text-white px-4 py-2 rounded-lg">
          <p className="text-sm">Connect to your external server using the SSH form.</p>
        </div>
      </div>

      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">Container Services</h2>
        <div className="grid grid-cols-2 gap-4">
          <button
            onClick={() => onServiceToggle('apache')}
            className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg flex items-center justify-center"
          >
            <Server className="mr-2" size={18} />
            Apache
          </button>
          <button
            onClick={() => onServiceToggle('squid')}
            className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg flex items-center justify-center"
          >
            <Server className="mr-2" size={18} />
            Squid
          </button>
        </div>
      </div>

      <div className="mt-auto">
        <details className="bg-gray-700 rounded-lg p-4">
          <summary className="font-semibold cursor-pointer">About Offensive Docker</summary>
          <div className="mt-4 text-sm space-y-2">
            <p>Offensive Docker includes numerous pentesting tools organized in categories:</p>
            <ul className="list-disc ml-4">
              <li>Port scanning: nmap, masscan, naabu</li>
              <li>Recon: Amass, GoBuster, Sublist3r</li>
              <li>Web Scanning: whatweb, wafw00z, nikto</li>
              <li>OWASP: sqlmap, XSStrike, jwt_tool</li>
              <li>Wordlists: SecList, dirb, wfuzz, rockyou</li>
            </ul>
          </div>
        </details>
      </div>
    </div>
  );
};
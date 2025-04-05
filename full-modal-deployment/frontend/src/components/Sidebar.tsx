import React, { useState } from 'react';
import { Server } from 'lucide-react';

interface SimplifiedSidebarProps {
  onServiceToggle?: (service: string) => void;
}

export const Sidebar: React.FC<SimplifiedSidebarProps> = ({
  onServiceToggle = () => {},
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  return (
    <div className="w-64 text-white p-4 flex flex-col h-screen border-r border-cyan">
      <h1 className="text-2xl font-bold text-cyan mb-24">AI Cyber Labs</h1>

      <div className="mb-8 border border-cyan rounded-lg p-3">
        <h2 className="text-xl font-semibold mb-4">Connect to External Container</h2>
        <h2 className="text-xl font-semibold text- mb-4">DarkCircuit</h2>
        <div className="bg-gray-700 text-white px-4 py-2 rounded-lg">
          <p className="text-sm">Connect to your external server using the SSH form.</p>
        </div>
      </div>

      <div className="mb-8 border border-cyan rounded-lg p-3">
        <h2 className="text-xl font-semibold mb-4">Container Services</h2>
        <div className="flex flex-col gap-3">
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
        <button
          onClick={openModal}
          className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg w-full text-center"
        >
          About Offensive Docker
        </button>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-gray-800 text-white p-6 rounded-lg w-96">
            <h2 className="text-xl font-bold mb-4">About Offensive Docker</h2>
            <p className="mb-4">
              Offensive Docker includes numerous pentesting tools organized in categories:
            </p>
            <ul className="list-disc ml-4 mb-4">
              <li>Port scanning: nmap, masscan, naabu</li>
              <li>Recon: Amass, GoBuster, Sublist3r</li>
              <li>Web Scanning: whatweb, wafw00z, nikto</li>
              <li>OWASP: sqlmap, XSStrike, jwt_tool</li>
              <li>Wordlists: SecList, dirb, wfuzz, rockyou</li>
            </ul>
            <button
              onClick={closeModal}
              className="bg-cyan hover:bg-bgCyan px-4 py-2 rounded-lg w-full text-black text-center"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
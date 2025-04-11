import React, { useState } from 'react';
import { Server, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface SimplifiedSidebarProps {
  onServiceToggle?: (service: string) => void;
}

export const Sidebar: React.FC<SimplifiedSidebarProps> = ({
  onServiceToggle = () => {},
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(true);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);
  const toggleSidebar = () => setIsCollapsed(!isCollapsed);

  return (
    <div
      className={`${
        isCollapsed ? 'w-14' : 'w-64'
      } transition-all duration-300 text-white p-4 flex flex-col h-screen border-r border-cyan relative`}
    >
      {/* Vertical rotated label when collapsed */}
      {isCollapsed && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rotate-90 origin-center text-cyan text-2xl font-MonomaniacOne whitespace-nowrap select-none">
          AI Cyber Labs
        </div>
      )}

      {/* Top: title & toggle */}
      <div className="flex items-center justify-between mb-8">
        {!isCollapsed && (
          <h1 className="text-3xl text-cyan font-MonomaniacOne select-none">AI Cyber Labs</h1>
        )}
        <button onClick={toggleSidebar} className="text-cyan hover:text-bgCyan">
          {isCollapsed ? <ChevronsRight size={20} /> : <ChevronsLeft size={20} />}
        </button>
      </div>

      {!isCollapsed && (
        <>
          <div className="mb-8 border border-cyan rounded-lg p-3">
            <h2 className="text-xl font-semibold mb-4">Connect to External Container</h2>
            <h2 className="text-xl font-semibold mb-4">DarkCircuit</h2>
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
              className="bg-cyan hover:bg-bgCyan px-4 py-2 rounded-lg w-full text-black text-center"
            >
              About Offensive Docker
            </button>
          </div>
        </>
      )}

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

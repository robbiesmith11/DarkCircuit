import React, { useState } from 'react';
import { Server, ChevronsLeft, ChevronsRight } from 'lucide-react';
import teamLogo from '../../src/components/AICyberLabs_Logo_Blue.png';

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
      {isCollapsed && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rotate-90 origin-center text-cyan text-2xl font-MonomaniacOne whitespace-nowrap select-none">
          Target Settings
        </div>
      )}

      {!isCollapsed && (
        <div className="mb-8">
          <img
            src={teamLogo}
            alt="AI Cyber Labs Logo"
            className="h-16 w-auto"
          />
        </div>
      )}

      <button
        onClick={toggleSidebar}
        className={`text-cyan hover:text-bgCyan absolute ${
          isCollapsed ? 'top-4 right-4' : 'top-1/2 -translate-y-1/2 right-4'
        }`}
      >
        {isCollapsed ? <ChevronsRight size={20} /> : <ChevronsLeft size={20} />}
      </button>

      {!isCollapsed && (
          <>
            <div className="mb-8 border border-cyan rounded-lg p-3">
              <h2 className="text-xl font-semibold mb-4">Container Services</h2>
              <div className="flex flex-col gap-3">
                <button
                    onClick={() => onServiceToggle('apache')}
                    className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg flex items-center justify-center"
                >
                  <Server className="mr-2" size={18}/>
                  Apache
                </button>
                <button
                    onClick={() => onServiceToggle('squid')}
                    className="bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded-lg flex items-center justify-center"
                >
                  <Server className="mr-2" size={18}/>
                  Squid
                </button>
              </div>
            </div>

            <div className="mt-auto">
              <button
                  onClick={openModal}
                  className="bg-cyan hover:bg-bgCyan px-4 py-2 rounded-lg w-full text-black text-center"
              >
                About Pwnbox
              </button>
            </div>
          </>
      )}

      {isModalOpen && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="bg-gray-800 text-white p-6 rounded-lg w-96">
                  <h2 className="text-xl font-bold mb-4">About Pwnbox</h2>
                  <p className="mb-4">
                      Pwnbox is Hack The Box’s cloud-based Parrot OS VM with 600+ pre-installed tools for penetration
                      testing:
                  </p>
                  <ul className="list-disc ml-4 mb-4">
                      <li>Enumeration: Nmap, Gobuster, theHarvester, Masscan</li>
                      <li>Exploitation: Metasploit, SQLMap, Burp Suite Community</li>
                      <li>Post-Exploitation: LinPEAS, BloodHound, Impacket</li>
                      <li>Reverse Engineering: Radare2, GDB, Cutter</li>
                      <li>Password Cracking: John, Hashcat, Hydra, SecLists</li>
                      <li>Web Testing: Nikto, Wfuzz, WPScan, OWASP ZAP</li>
                      <li>Wireless: Aircrack-ng, Reaver, Wifite, Bettercap</li>
                      <li>Forensics: Autopsy, Volatility, Binwalk, ExifTool</li>
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
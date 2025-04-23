import React, { useState, useEffect, useCallback } from 'react';
import { Info, ChevronsLeft, ChevronsRight } from 'lucide-react';
import teamLogo from '../../src/components/AICyberLabs_Logo_Blue.png';

// Define the HTB challenge list
const HTB_CHALLENGES = [
  "Dancing",
  "Fawn",
  "Lame"
];

interface SimplifiedSidebarProps {
  selectedChallenge?: string;
  targetIp?: string;
  onChallengeSelect?: (challenge: string) => void;
  onTargetIpChange?: (ip: string) => void;
}

export const Sidebar: React.FC<SimplifiedSidebarProps> = ({
  selectedChallenge = '',
  targetIp = '',
  onChallengeSelect = () => {},
  onTargetIpChange = () => {}
}) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [localTargetIp, setLocalTargetIp] = useState(targetIp);

  // Update local IP when prop changes
  useEffect(() => {
    if (targetIp !== localTargetIp) {
      setLocalTargetIp(targetIp);
    }
  }, [targetIp]);

  const openModal = useCallback(() => setIsModalOpen(true), []);
  const closeModal = useCallback(() => setIsModalOpen(false), []);
  const toggleSidebar = useCallback(() => setIsCollapsed(!isCollapsed), [isCollapsed]);

  // Update challenge handling
  const handleChallengeChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const newChallenge = e.target.value;
    console.log("Sidebar - challenge changed:", newChallenge);
    onChallengeSelect(newChallenge);
  }, [onChallengeSelect]);

  // Update the target IP handling
  const handleIpChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalTargetIp(e.target.value);
  }, []);

  // Important: This is where we update the parent's state
  const handleIpBlur = useCallback(() => {
    console.log("Sidebar - IP blur, sending value:", localTargetIp);
    // Always update the parent to ensure consistent state
    onTargetIpChange(localTargetIp);
  }, [localTargetIp, onTargetIpChange]);

  return (
    <div
      className={`${
        isCollapsed ? 'w-14' : 'w-64'
      } transition-all duration-300 text-white p-4 flex flex-col h-screen border-r border-cyan relative`}
    >
      {isCollapsed && (
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rotate-90 origin-center text-cyan text-2xl font-MonomaniacOne whitespace-nowrap select-none">
          Target Configuration
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
              <div className="flex items-center justify-between mb-2">
                <h2 className="text-xl font-bold text-white">Target Configuration</h2>
                <div className="relative group">
                  <Info size={16} className="text-cyan cursor-help"/>
                  <div
                      className="absolute left-0 bottom-full mb-2 w-64 bg-gray-800 text-white p-2 rounded-md text-sm opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                    This information will be automatically appended to your first message in a conversation to provide
                    context to the assistant.
                  </div>
                </div>
              </div>
              <div className="flex flex-col gap-3">
                {/* HTB Challenge Dropdown */}
                <div className="flex flex-col gap-1">
                  <label htmlFor="htb-challenge" className="text-sm text-gray-300">
                    HackTheBox Challenge
                  </label>
                  <select
                      id="htb-challenge"
                      value={selectedChallenge}
                      onChange={handleChallengeChange}
                      className="bg-gray-700 text-white px-4 py-2 rounded-lg w-full"
                  >
                    <option value="">Select a challenge</option>
                    {HTB_CHALLENGES.map(challenge => (
                        <option key={challenge} value={challenge}>{challenge}</option>
                    ))}
                  </select>
                </div>

                {/* Target IP Input */}
                <div className="flex flex-col gap-1">
                  <label htmlFor="target-ip" className="text-sm text-gray-300">
                    Target IP
                  </label>
                  <input
                      id="target-ip"
                      type="text"
                      value={localTargetIp}
                      onChange={handleIpChange}
                      onBlur={handleIpBlur}
                      // Also update on Enter key
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleIpBlur();
                          // Blur the input to hide keyboard on mobile
                          (e.target as HTMLInputElement).blur();
                        }
                      }}
                      placeholder="e.g. 10.129.203.101"
                      className="bg-gray-700 text-white px-4 py-2 rounded-lg w-full"
                  />
                </div>
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
                    Pwnbox is Hack The Box's cloud-based Parrot OS VM with 600+ pre-installed tools for penetration
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
import React, { useState, useRef, useEffect } from 'react';
import { GripVertical, BarChart3, Users, TrendingUp, Activity } from 'lucide-react';

const ResizablePanel = ({ children, minWidth = 200, initialWidth = 300, side = 'left' }) => {
  const [width, setWidth] = useState(initialWidth);
  const [isResizing, setIsResizing] = useState(false);
  const panelRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing) return;

      if (panelRef.current) {
        const rect = panelRef.current.getBoundingClientRect();
        let newWidth;
        
        if (side === 'left') {
          newWidth = e.clientX - rect.left;
        } else {
          newWidth = rect.right - e.clientX;
        }
        
        if (newWidth >= minWidth) {
          setWidth(newWidth);
        }
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, minWidth, side]);

  return (
    <div
      ref={panelRef}
      className="relative flex-shrink-0"
      style={{ width: `${width}px` }}
    >
      {children}
      <div
        className={`absolute top-0 ${side === 'left' ? 'right-0' : 'left-0'} h-full w-1 cursor-col-resize hover:bg-blue-500 transition-colors ${
          isResizing ? 'bg-blue-500' : 'bg-gray-300'
        }`}
        onMouseDown={() => setIsResizing(true)}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 left-1/2 bg-gray-400 rounded p-1 opacity-0 hover:opacity-100 transition-opacity">
          <GripVertical size={16} />
        </div>
      </div>
    </div>
  );
};

const DashboardCard = ({ title, value, icon: Icon, color }) => (
  <div className="bg-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-gray-600 text-sm font-medium">{title}</h3>
      <Icon className={`${color}`} size={20} />
    </div>
    <p className="text-2xl font-bold text-gray-900">{value}</p>
  </div>
);

export default function ResizableDashboard() {
  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white shadow-sm px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">Resizable Dashboard</h1>
        <p className="text-sm text-gray-600 mt-1">Drag the dividers to resize panels</p>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <ResizablePanel initialWidth={250} minWidth={200} side="left">
          <div className="h-full bg-white border-r border-gray-200 p-4 overflow-y-auto">
            <h2 className="text-lg font-semibold mb-4 text-gray-900">Navigation</h2>
            <nav className="space-y-2">
              {['Dashboard', 'Analytics', 'Reports', 'Settings', 'Users', 'Help'].map((item) => (
                <button
                  key={item}
                  className="w-full text-left px-3 py-2 rounded hover:bg-gray-100 text-gray-700 transition-colors"
                >
                  {item}
                </button>
              ))}
            </nav>
          </div>
        </ResizablePanel>

        {/* Center Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 p-6 overflow-y-auto">
            <div className="max-w-6xl mx-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <DashboardCard 
                  title="Total Users" 
                  value="12,543" 
                  icon={Users}
                  color="text-blue-600"
                />
                <DashboardCard 
                  title="Revenue" 
                  value="$45,231" 
                  icon={TrendingUp}
                  color="text-green-600"
                />
                <DashboardCard 
                  title="Active Now" 
                  value="3,421" 
                  icon={Activity}
                  color="text-purple-600"
                />
                <DashboardCard 
                  title="Conversions" 
                  value="892" 
                  icon={BarChart3}
                  color="text-orange-600"
                />
              </div>

              <div className="bg-white rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold mb-4 text-gray-900">Recent Activity</h2>
                <div className="space-y-3">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-0">
                      <div>
                        <p className="font-medium text-gray-900">Activity Item {i}</p>
                        <p className="text-sm text-gray-600">Description of the activity</p>
                      </div>
                      <span className="text-sm text-gray-500">{i}h ago</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Sidebar */}
        <ResizablePanel initialWidth={300} minWidth={250} side="right">
          <div className="h-full bg-white border-l border-gray-200 p-4 overflow-y-auto">
            <h2 className="text-lg font-semibold mb-4 text-gray-900">Details Panel</h2>
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 mb-2">Quick Stats</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Completion Rate</span>
                    <span className="font-medium">87%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg. Response Time</span>
                    <span className="font-medium">2.3s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Satisfaction</span>
                    <span className="font-medium">4.8/5</span>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 mb-2">Notifications</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <p>• New user registered</p>
                  <p>• Report generated</p>
                  <p>• System update available</p>
                </div>
              </div>
            </div>
          </div>
        </ResizablePanel>
      </div>
    </div>
  );
}
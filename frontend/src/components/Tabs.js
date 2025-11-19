import { useState } from 'react';

export default function Tabs() {
  const [activeTab, setActiveTab] = useState(0);

  const tabs = [
    { label: 'Profile', content: 'Profile information goes here.' },
    { label: 'Settings', content: 'Settings and preferences.' },
    { label: 'Messages', content: 'Your messages appear here.' }
  ];

  return (
    <div style={styles.container}>
      {/* Tab Headers */}
      <div style={styles.tabHeaders}>
        {tabs.map((tab, index) => (
          <button
            key={index}
            onClick={() => setActiveTab(index)}
            style={{
              ...styles.tabButton,
              ...(activeTab === index ? styles.tabButtonActive : {})
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div style={styles.tabContent}>
        <h2 style={styles.contentTitle}>{tabs[activeTab].label}</h2>
        <p style={styles.contentText}>{tabs[activeTab].content}</p>
      </div>
    </div>
  );
}

const styles = {
  container: {
    width: '80%',
    maxWidth: '672px',
    margin: '0 auto',
    padding: '8px'
  },
  tabHeaders: {
    display: 'flex',
    borderBottom: '2px solid #d1d5db'
  },
  tabButton: {
    padding: '4px 10px',
    fontWeight: '500',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    color: '#4b5563',
    borderBottom: '2px solid transparent',
    marginBottom: '-2px',
    transition: 'color 0.2s'
  },
  tabButtonActive: {
    color: '#2563eb',
    borderBottom: '2px solid #2563eb'
  },
  tabContent: {
    padding: '20px',
    border: '1px solid #d1d5db',
    borderTop: 'none',
    background: 'white'
  },
  contentTitle: {
    fontSize: '20px',
    fontWeight: '600',
    marginBottom: '16px'
  },
  contentText: {
    color: '#374151'
  }
};

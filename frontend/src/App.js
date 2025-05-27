import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell, Legend, ResponsiveContainer } from 'recharts';

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [chartData, setChartData] = useState(null);
  const [chartType, setChartType] = useState(null);

  const handleSubmit = async () => {
    setAnswer("Thinking...");
    const res = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    setAnswer(data.result);
    if (data.chart) {
      setChartData(data.chart.data);
      setChartType(data.chart.type);
    } else {
      setChartData(null);
      setChartType(null);
    }
  };

  const renderChart = () => {
    if (!chartData || !chartType) return null;
    const data = Object.entries(chartData).map(([name, value]) => ({ name, value }));
    if (chartType === "bar") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </ResponsiveContainer>
      );
    }
    if (chartType === "pie") {
      return (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" label>
              {data.map((entry, idx) => <Cell key={idx} />)}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      );
    }
    return null;
  };

  return (
    <div style={{ padding: 20 }}>
      <header style={{ marginBottom: 20, display: 'flex', alignItems: 'center' }}>
        <img src="/logo.png" alt="Logo" style={{ height: 40, marginRight: 10 }} />
        <span style={{ fontSize: 24, fontWeight: "bold" }}>Marketing Analytics Demo</span>
      </header>
      <div style={{ display: "flex", marginBottom: 20 }}>
        <input
          style={{ flexGrow: 1, padding: 10, fontSize: 16 }}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a marketing question..."
        />
        <button
          style={{ backgroundColor: "orange", color: "white", padding: "10px 20px", border: "none", cursor: "pointer" }}
          onClick={handleSubmit}
        >Execute</button>
      </div>
      <div style={{ whiteSpace: "pre-wrap", marginBottom: 20 }}>{answer}</div>
      {renderChart()}
    </div>
  );
}

export default App;

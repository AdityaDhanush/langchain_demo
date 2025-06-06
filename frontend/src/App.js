// App.js
import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ResponsiveContainer,
  BarChart, Bar,
  PieChart, Pie, Cell,
  LineChart, Line,
  XAxis, YAxis, Tooltip
} from 'recharts';
import {
  Container,
  Box,
  Paper,
  TextField,
  Button,
  Typography
} from '@mui/material';

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [chartData, setChartData] = useState(null);
  const [chartType, setChartType] = useState(null);

  const handleSubmit = async () => {
    if (!query.trim()) return;
    setAnswer("Thinking...");
    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      let answerText;
      if (data.result && typeof data.result === "object") {
        answerText = data.result.output ?? JSON.stringify(data.result, null, 2);
      } else {
        answerText = data.result || data.summary || JSON.stringify(data);
      }
      setAnswer(answerText);

      if (data.chart) {
        setChartData(data.chart.data);
        setChartType(data.chart.type);
      } else {
        setChartData(null);
        setChartType(null);
      }
    } catch (err) {
      setAnswer("Error fetching response.");
      console.error(err);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  const renderChart = () => {
    if (!chartData || !chartType) return null;
    const data = Object.entries(chartData).map(([name, value]) => ({ name, value }));
    switch (chartType) {
      case "bar":
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
      case "pie":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={data} dataKey="value" nameKey="name" label>
                {data.map((entry, idx) => <Cell key={idx} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        );
      case "line":
        return (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="value" />
            </LineChart>
          </ResponsiveContainer>
        );
      default:
        return null;
    }
  };

  const renderAnswer = () => {
    if (!answer) return null;
    // Turn literal "\n" into real newlines
    const text = answer.replace(/\\n/g, '\n');
    return (
      <Box
        component={Paper}
        elevation={2}
        sx={{ p: 3, mb: 4, borderRadius: '20px', bgcolor: 'background.paper' }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({node, ...props}) => <Typography variant="body1" paragraph {...props} />,
            li: ({node, ...props}) => <Typography component="li" variant="body1" {...props} />,
            ul: ({node, ...props}) => <Box component="ul" sx={{pl:4, mb:2}} {...props} />,
          }}
        >
          {text}
        </ReactMarkdown>
      </Box>
    );
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" alignItems="center" mb={4}>
        <img
          src="/netelixir-logo.svg"
          alt="Logo"
          style={{ height: 48, marginRight: 12 }}
        />
        <Typography variant="h4" component="h1">
          Marketing Analytics Langchain Demo
        </Typography>
      </Box>

      {/* Query input panel */}
      <Paper
        elevation={4}
        sx={{
          p: 2,
          mb: 4,
          display: 'flex',
          alignItems: 'center',
          borderRadius: '38px',
          boxShadow: theme => theme.shadows[4],
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder="e.g. What is the monthly AOV trend for the last year?"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          sx={{
            bgcolor: 'background.paper',
            '& .MuiOutlinedInput-root': {
              borderRadius: '30px',
            },
            mr: 2,
          }}
        />
        <Button
          variant="contained"
          size="medium"
          onClick={handleSubmit}
          sx={{
            bgcolor: '#f58120',
            color: '#fff',
            px: 3,
            py: 1.5,
            borderRadius: '38px',
            boxShadow: theme => theme.shadows[2],
            transition: 'transform 0.1s ease-in-out, box-shadow 0.2s',
            '&:hover': {
              boxShadow: theme => theme.shadows[6],
            },
            '&:active': {
              transform: 'scale(0.95)',
            }
          }}
        >
          Ask Insights
        </Button>
      </Paper>

      {/* Answer */}
      {renderAnswer()}

      {/* Chart */}
      {renderChart()}
    </Container>
  );
}

export default App;

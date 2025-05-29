// App.js
import React, { useState } from 'react';
import { Fragment } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ResponsiveContainer,
  BarChart, Bar,
  PieChart, Pie, Cell,
  LineChart, Line,
  XAxis, YAxis, Tooltip
} from 'recharts';

// MUI imports
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
    const res = await fetch("http://localhost:8000/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    // unwrap answer
    let answerText;
    if (data.result && typeof data.result === "object") {
      answerText = data.result.output ?? JSON.stringify(data.result, null, 2);
    } else {
      answerText = data.result || data.summary || JSON.stringify(data);
    }
    setAnswer(answerText);

    // chart
    if (data.chart) {
      setChartData(data.chart.data);
      setChartType(data.chart.type);
    } else {
      setChartData(null);
      setChartType(null);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };
// Helper: take a string, split on **bold**, and return array of React nodes
   const renderInline = (text) => {
     const parts = text.split(/(\*\*[^*] \*\*)/g).filter(Boolean);
     return parts.map((part, i) => {
       const match = part.match(/^\*\*(. )\*\*$/);
       if (match) {
         return (
           <Typography
             component="span"
             variant="body1"
             key={i}
             sx={{ fontWeight: 'bold' }}
           >
             {match[1]}
           </Typography>
         );
       }
       return (
         <Typography component="span" variant="body1" key={i}>
           {part}
         </Typography>
       );
     });
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

// Render the LLM’s summary with real paragraphs and lists
//  const renderAnswer = () => {
//    if (!answer) return null;
//    // split on newlines
////    const lines = answer.split('\n');
//    // convert literal "\n" into real newlines, then split
//    const text = answer.replace(/\\n/g, '\n');
//    const lines = text.split('\n');
//    const elements = [];
//    let listItems = [];
//
//     const flushList = () => {
//       if (listItems.length > 0) {
//         elements.push(
//           <Box component="ul" key={`ul-${elements.length}`} sx={{ pl: 4, mb: 2 }}>
//             {listItems.map((item, i) => (
//              <Typography
//                component="li"
//                variant="body1"
//                key={`li-${i}`}
//                sx={{ mb: 0.5 }}
//              >
//                {renderInline(item)}
//              </Typography>
//            ))}
//           </Box>
//         );
//         listItems = [];
//       }
//     };
//
//     lines.forEach((raw, idx) => {
//       const line = raw.trim();
//       if (!line) {
//         // blank line: break current list and add spacing
//         flushList();
//         elements.push(<Box key={`br-${idx}`} sx={{ height: 8 }} />);
//       } else if (/^[-*]\s /.test(line)) {
//         // bullet line
//         listItems.push(line.replace(/^[-*]\s /, ''));
//       } else {
//         // normal text: flush any pending list, then add paragraph
//         flushList();
//         elements.push(
//           <Typography variant="body1" paragraph key={`p-${idx}`}>
//             {renderInline(line)}
//           </Typography>
//         );
//       }
//     });
//     // flush at end
//     flushList();
//     return elements;
//   };

  // Turn literal “\n” into real linebreaks, then render markdown
  const renderAnswer = () => {
    if (!answer) return null;
    const text = answer.replace(/\\n/g, '\n');
    return (
      <Box
        component={Paper}
        elevation={2}
        sx={{ p: 3, mb: 4, borderRadius: '38px', bgcolor: 'background.paper' }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Use MUI Typography for paragraphs
            p: ({node, ...props}) => <Typography variant="body1" paragraph {...props} />,
            // Use MUI Typography for list items
            li: ({node, ...props}) => <Typography component="li" variant="body1" {...props} />,
            // Render unordered lists with proper padding
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
      {/* …header, input panel… */}
      {renderAnswer()}
      {/* …chart… */}
    </Container>
  );
}
}
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
          borderRadius: 20,
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
            '& .MuiOutlinedInput-root' : {
            borderRadius: 20,
            },
            mr: 2,
          }}
        />
        <Button
          variant="contained"
          size="medium"
          onClick={handleSubmit}
          onKeyDown={onKeyDown}
          sx={{
            bgcolor: '#f58120',
            color: '#fff',
            px: 3,
            py: 1.5,
            borderRadius: 38,
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
      <Paper
        elevation={2}
        sx={{
          p: 3,
          mb: 4,
          borderRadius: '15px',
          bgcolor: 'background.paper'
        }}
      >
        {renderAnswer()}
      </Paper>


      {/* Chart */}
      {renderChart()}
    </Container>
  );
}

export default App;

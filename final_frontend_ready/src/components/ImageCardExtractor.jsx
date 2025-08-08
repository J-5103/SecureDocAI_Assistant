import React, { useState } from 'react';

function ImageCardExtractor() {
  const [frontImage, setFrontImage] = useState<File | null>(null);
  const [backImage, setBackImage] = useState<File | null>(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!frontImage) {
      setError('Please select the front image.');
      return;
    }

    setError(null);
    setResult('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('front_image', frontImage);
      if (backImage) formData.append('back_image', backImage);

      const response = await fetch('192.168.0.109:8000/api/ask-image', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.status === 'success') {
        setResult(data.data);
      } else {
        setError(data.message || 'Something went wrong.');
      }
    } catch (err) {
      setError('Error uploading image or connecting to server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h2>🖼️ Business Card Extractor</h2>

      <label style={styles.label}>Front Image (required):</label>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFrontImage(e.target.files?.[0] || null)}
      />

      <label style={styles.label}>Back Image (optional):</label>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setBackImage(e.target.files?.[0] || null)}
      />

      <button onClick={handleSubmit} style={styles.button} disabled={loading}>
        {loading ? 'Processing...' : 'Extract Info'}
      </button>

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.result}>
          <h4>✅ Extracted Data:</h4>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre>
        </div>
      )}
    </div>
  );
}

// 👇 Just normal JS object with inferred types
const styles = {
  container: {
    maxWidth: '600px',
    margin: '40px auto',
    padding: '20px',
    border: '1px solid #ccc',
    borderRadius: '8px',
    backgroundColor: '#f9f9f9',
    fontFamily: 'sans-serif',
  },
  label: {
    display: 'block',
    marginTop: '12px',
    fontWeight: 'bold',
  },
  button: {
    marginTop: '20px',
    padding: '10px 20px',
    backgroundColor: '#004d40',
    color: '#fff',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
  },
  error: {
    color: 'red',
    marginTop: '12px',
  },
  result: {
    marginTop: '30px',
    background: '#fff',
    padding: '15px',
    border: '1px solid #ccc',
    borderRadius: '5px',
  },
};

export default ImageCardExtractor;

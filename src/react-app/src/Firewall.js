// Firewall.js
import React, { Component } from 'react';
import axios from 'axios';

class Firewall extends Component {
  state = {
    data: null,
    error: null
  }

  componentDidMount() {
    axios.post('/upload_file')
      .then(response => {
        this.setState({ data: response.data })
      })
      .catch(error => {
        this.setState({ error: error.message })
      })
  }

  render() {
    const { data, error } = this.state;

    if (error) {
      return <div>{error}</div>;
    }

    if (!data) {
      return <div>Loading...</div>;
    }

    return (
      <div>
        <h1>{data.filename}</h1>
        <div dangerouslySetInnerHTML={{ __html: data.tables[0] }} />
      </div>
    );
  }
}

export default Firewall;
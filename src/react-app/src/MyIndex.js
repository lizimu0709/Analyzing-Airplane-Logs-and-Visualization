// MyIndex.js
import React from 'react';
import { Link } from 'react-router-dom';


class MyIndex extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      html: null
    };
  }

  componentDidMount() {
      fetch('index.html')
      .then(response => response.text())
      .then(html => {
        console.log(html);
        this.setState({ html });
      })
      .catch(error => {
        console.log('Error fetching index.html:', error);
        this.setState({ html: '<p>Error fetching index.html.</p>' });
      });
  }

  render() {
    return (
      <div dangerouslySetInnerHTML={{ __html: this.state.html }}></div>
    );
  }
}

export default MyIndex;

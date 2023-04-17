// Dataload.js
import React from 'react';

class Dataload extends React.Component {
    constructor(props) {
      super(props);
      this.state = {
        html: null
      };
    }
  
    componentDidMount() {
        fetch('dataload.html')
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
  
  export default Dataload;
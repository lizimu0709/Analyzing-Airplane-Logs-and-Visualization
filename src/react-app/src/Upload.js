import React from 'react';

class Upload extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      file: null,
      error: null
    };
  }

  handleFileChange = (event) => {
    this.setState({
      file: event.target.files[0]
    });
  }

  handleFormSubmit = (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append('inputFile', this.state.file);
    fetch('/upload_file', {
      method: 'POST',
      body: formData
    })
    .then(response => response.text())
    .then(data => {
      // Do something with the response data
    })
    .catch(error => {
      this.setState({
        error: 'An error occurred while uploading the file.'
      });
    });
  }

  render() {
    return (
      <div>
        <form onSubmit={this.handleFormSubmit}>
          <input type="file" name="inputFile" onChange={this.handleFileChange} />
          <input type="submit" value="Upload" />
        </form>
        {this.state.error && <p>{this.state.error}</p>}
      </div>
    );
  }
}

export default Upload;

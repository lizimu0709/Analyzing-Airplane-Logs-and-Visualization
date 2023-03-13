-- Create the Airplanes table with the airplane_id as the primary key
CREATE TABLE Airplanes (
  airplane_id INT PRIMARY KEY AUTO INCREMENT,
  tail_number VARCHAR(255) NOT NULL,
  episode_start_date DATETIME NOT NULL
);

-- Create the Dataloads table with the dataload_id as the primary key
CREATE TABLE Dataloads (
  dataload_id INT PRIMARY KEY AUTO INCREMENT,
  dataload_target VARCHAR(255),
  dataload_protocol VARCHAR(255),
);

-- Create the Software table with the software_id as the primary key
CREATE TABLE Softwares (
  software_id INT PRIMARY KEY AUTO INCREMENT,
  crate_number VARCHAR(255) NOT NULL,
  part_number VARCHAR(255),
);

-- Create the Firewall table with the firewall_id as the primary key
CREATE TABLE Firewalls (
  firewall_id INT PRIMARY KEY AUTO INCREMENT,
  source_ip_address VARCHAR(15) NOT NULL,
  destination_ip_address VARCHAR(15) NOT NULL,
  proto VARCHAR(15) NOT NULL,
  spt INT NOT NULL,
  dpt INT NOT NULL,
);

-- Create the Features table with the feature_id as the primary key
CREATE TABLE Features (
  feature_id INT PRIMARY KEY AUTO INCREMENT,
  feature_name VARCHAR(255) NOT NULL,
  episode_id INT NOT NULL REFERENCES Episodes(episode_id)
);

-- Create the Features table with the episode_id as the primary key
CREATE TABLE Episodes (
    episode_id INT PRIMARY KEY AUTO INCREMENT,
    airplane_id INT NOT NULL REFERENCES Airplanes (airplane_id),
    dataload_id INT NOT NULL REFERENCES Dataloads (dataload_id),
    software_id INT NOT NULL REFERENCES Softwares (software_id),
    firewall_id INT NOT NULL REFERENCES Firewalls (firewall_id)
);



-- Create the Airplanes table with the airplane_id as the primary key
CREATE TABLE Airplanes (
  airplane_id INT IDENTITY(1,1) PRIMARY KEY,
  tail_number VARCHAR(255) NOT NULL,
  episode_start_date DATETIME NOT NULL
);

-- Create the Dataloads table with the dataload_id as the primary key
CREATE TABLE Dataloads (
  dataload_id INT IDENTITY(1,1) PRIMARY KEY,
  dataload_target VARCHAR(255),
  dataload_protocol VARCHAR(255),
);

-- Create the Software table with the software_id as the primary key
CREATE TABLE Softwares (
  software_id INT IDENTITY(1,1) PRIMARY KEY,
  crate_number VARCHAR(255) NOT NULL,
  part_number VARCHAR(255),
);

-- Create the Firewall table with the firewall_id as the primary key
CREATE TABLE Firewalls (
  firewall_id INT IDENTITY(1,1) PRIMARY KEY,
  source_ip_address VARCHAR(15) NOT NULL,
  destination_ip_address VARCHAR(15) NOT NULL,
  proto VARCHAR(15) NOT NULL,
  spt INT NOT NULL,
  dpt INT NOT NULL,
);

-- Create the Features table with the feature_id as the primary key
CREATE TABLE Features (
  feature_id INT IDENTITY(1,1) PRIMARY KEY,
  feature_name VARCHAR(255) NOT NULL,
  episode_id INT NOT NULL REFERENCES Episodes(episode_id)
);

-- Create the Features table with the episode_id as the primary key
CREATE TABLE Episodes (
    episode_id INT IDENTITY(1,1) PRIMARY KEY,
    airplane_id INT NOT NULL REFERENCES Airplanes (airplane_id),
    dataload_id INT NOT NULL REFERENCES Dataloads (dataload_id),
    software_id INT NOT NULL REFERENCES Softwares (software_id),
    firewall_id INT NOT NULL REFERENCES Firewalls (firewall_id)
);



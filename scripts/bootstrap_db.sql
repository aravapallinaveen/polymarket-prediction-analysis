-- Run once to set up the RDS database
CREATE DATABASE polymarket_analysis;
\c polymarket_analysis;

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE polymarket_analysis TO admin;

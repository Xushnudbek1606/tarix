-- GovSign PostgreSQL schema (generated from backend/app/models.py).
-- The application creates missing tables automatically on start-up; use this file for DBA review / manual provisioning.
-- Personal fields (citizens.full_name, messages.text) hold Fernet ciphertext; face_embeddings.vector_enc is encrypted; pins are bcrypt hashes.

CREATE TABLE audit_logs (
	id SERIAL NOT NULL, 
	ts TIMESTAMP WITH TIME ZONE NOT NULL, 
	actor VARCHAR(80) NOT NULL, 
	action VARCHAR(64) NOT NULL, 
	target VARCHAR(120) NOT NULL, 
	detail VARCHAR(500) NOT NULL, 
	ip VARCHAR(64) NOT NULL, 
	PRIMARY KEY (id)
);

CREATE INDEX ix_audit_logs_action ON audit_logs (action);
CREATE INDEX ix_audit_logs_ts ON audit_logs (ts);

CREATE TABLE citizens (
	id SERIAL NOT NULL, 
	code VARCHAR(24) NOT NULL, 
	full_name TEXT NOT NULL, 
	national_id_hash VARCHAR(64), 
	pin_hash VARCHAR(128), 
	failed_pins INTEGER NOT NULL, 
	pin_locked_until TIMESTAMP WITH TIME ZONE, 
	script VARCHAR(10) NOT NULL, 
	active BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (national_id_hash)
);

CREATE UNIQUE INDEX ix_citizens_code ON citizens (code);

CREATE TABLE desks (
	id SERIAL NOT NULL, 
	code VARCHAR(32) NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	camera_index INTEGER NOT NULL, 
	camera_enabled BOOLEAN NOT NULL, 
	active BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (code)
);


CREATE TABLE model_versions (
	id SERIAL NOT NULL, 
	version VARCHAR(64) NOT NULL, 
	arch VARCHAR(32) NOT NULL, 
	num_classes INTEGER NOT NULL, 
	synthetic BOOLEAN NOT NULL, 
	val_accuracy FLOAT, 
	test_accuracy FLOAT, 
	path VARCHAR(300) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (version)
);


CREATE TABLE settings (
	key VARCHAR(64) NOT NULL, 
	value VARCHAR(200) NOT NULL, 
	PRIMARY KEY (key)
);


CREATE TABLE signs (
	id SERIAL NOT NULL, 
	name VARCHAR(64) NOT NULL, 
	uz_latin VARCHAR(120) NOT NULL, 
	uz_cyrillic VARCHAR(120) NOT NULL, 
	category VARCHAR(40) NOT NULL, 
	pos VARCHAR(16) NOT NULL, 
	active BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_signs_name ON signs (name);

CREATE TABLE users (
	id SERIAL NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	password_hash VARCHAR(128) NOT NULL, 
	role VARCHAR(16) NOT NULL, 
	full_name VARCHAR(160) NOT NULL, 
	active BOOLEAN NOT NULL, 
	failed_logins INTEGER NOT NULL, 
	locked_until TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_users_username ON users (username);

CREATE TABLE employees (
	id SERIAL NOT NULL, 
	user_id INTEGER NOT NULL, 
	desk_id INTEGER, 
	position VARCHAR(120) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (user_id), 
	FOREIGN KEY(user_id) REFERENCES users (id), 
	FOREIGN KEY(desk_id) REFERENCES desks (id)
);


CREATE TABLE face_embeddings (
	id SERIAL NOT NULL, 
	citizen_id INTEGER NOT NULL, 
	embedder VARCHAR(64) NOT NULL, 
	dim INTEGER NOT NULL, 
	vector_enc BYTEA NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(citizen_id) REFERENCES citizens (id) ON DELETE CASCADE
);


CREATE TABLE sessions (
	id SERIAL NOT NULL, 
	service_number VARCHAR(24) NOT NULL, 
	citizen_id INTEGER, 
	employee_id INTEGER, 
	desk_id INTEGER NOT NULL, 
	auth_method VARCHAR(16) NOT NULL, 
	status VARCHAR(12) NOT NULL, 
	end_reason VARCHAR(32), 
	start_time TIMESTAMP WITH TIME ZONE NOT NULL, 
	end_time TIMESTAMP WITH TIME ZONE, 
	last_activity TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(citizen_id) REFERENCES citizens (id), 
	FOREIGN KEY(employee_id) REFERENCES users (id), 
	FOREIGN KEY(desk_id) REFERENCES desks (id)
);

CREATE INDEX ix_sessions_status ON sessions (status);

CREATE TABLE sign_datasets (
	id SERIAL NOT NULL, 
	sign_id INTEGER NOT NULL, 
	speaker_id VARCHAR(64) NOT NULL, 
	language VARCHAR(16) NOT NULL, 
	video_path VARCHAR(300), 
	landmarks_path VARCHAR(300) NOT NULL, 
	frames INTEGER NOT NULL, 
	translation VARCHAR(200) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(sign_id) REFERENCES signs (id)
);

CREATE INDEX ix_sign_datasets_sign_id ON sign_datasets (sign_id);

CREATE TABLE messages (
	id SERIAL NOT NULL, 
	session_id INTEGER NOT NULL, 
	sender VARCHAR(10) NOT NULL, 
	text TEXT NOT NULL, 
	lang VARCHAR(4) NOT NULL, 
	signs JSON, 
	confidence FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE INDEX ix_messages_session_id ON messages (session_id);

CREATE TABLE recognition_logs (
	id SERIAL NOT NULL, 
	session_id INTEGER, 
	desk_id INTEGER, 
	sign_name VARCHAR(64), 
	confidence FLOAT NOT NULL, 
	accepted BOOLEAN NOT NULL, 
	model_version VARCHAR(64) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(session_id) REFERENCES sessions (id), 
	FOREIGN KEY(desk_id) REFERENCES desks (id)
);

CREATE INDEX ix_recognition_logs_created_at ON recognition_logs (created_at);
CREATE INDEX ix_recognition_logs_session_id ON recognition_logs (session_id);

-- Least-privilege application role (example)
-- CREATE ROLE govsign_app LOGIN PASSWORD '<strong password>';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO govsign_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO govsign_app;
-- Audit logs should be append-only for the app role:
-- REVOKE UPDATE, DELETE ON audit_logs FROM govsign_app;

import sqlite3
import json

class SQLiteDatabase:
    def __init__(self, db_name="database.db"):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()
        self._create_table()
    
    def _create_table(self):
        """Creates a table if it doesn't exist."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS activations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration TEXT,
                image TEXT,
                container_id TEXT,
                json_data TEXT,
                experiment_id TEXT,
                func TEXT
            )
        ''')
        self.connection.commit()

    
    def save_data(self, activation_id=None, timestamp=None, duration=None, 
              image=None, container_id=None, json_obj=None, 
              experiment_id=None, func=None,):
        """Saves JSON data to the database, allowing optional fields."""
        
        json_str = json.dumps(json_obj) if json_obj else None  # Convert JSON only if provided
        
        # Prepare dynamic SQL query
        columns = ["activation_id", "timestamp", "duration", "image", 
                "container_id", "json_data", "experiment_id", "func"]
        
        values = [activation_id, timestamp, duration, image, 
                container_id, json_str, experiment_id, func]
        
        # Filter out None values
        columns = [col for col, val in zip(columns, values) if val is not None]
        values = [val for val in values if val is not None]
        
        # Construct dynamic SQL query
        sql = f"""
            INSERT INTO activations ({', '.join(columns)}) 
            VALUES ({', '.join(['?'] * len(values))})
        """
        
        # Execute the query
        self.cursor.execute(sql, values)
        self.connection.commit()

    
    def read_data(self, record_id=None):
        """Reads JSON data from the database. If no ID is provided, returns all records."""
        if record_id:
            self.cursor.execute("SELECT * FROM activations WHERE id = ?", (record_id,))
            result = self.cursor.fetchone()
            return self._format_result(result) if result else None
        else:
            self.cursor.execute("SELECT * FROM activations")
            return [self._format_result(row) for row in self.cursor.fetchall()]
    
    def update_data(self, record_id, update_fields):
        """Updates JSON data in the database."""
        set_clause = ", ".join([f"{key} = ?" for key in update_fields.keys()])
        values = list(update_fields.values()) + [record_id]
        
        self.cursor.execute(f"UPDATE activations SET {set_clause} WHERE id = ?", values)
        self.connection.commit()
    
    def _format_result(self, row):
        """Formats a database row into a dictionary."""
        return {
            "id": row[0],
            "activation_id": row[1],
            "timestamp": row[2],
            "duration": row[3],
            "image": row[4],
            "container_id": row[5],
            "json_data": json.loads(row[6]) if row[6] else None,
            "experiment_id": row[7]
        }
    
    def close(self):
        """Closes the database connection."""
        self.connection.close()
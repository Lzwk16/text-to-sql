### Task
Generate one SQLite SELECT query to answer [QUESTION]{question}[/QUESTION].
The query must be logically correct in all cases implied by the question, including situations where eligibility of records depends on aggregated properties of groups and where selecting a fixed number of records applies independently within each group.

### Rules
- Output ONLY the SQL query. No markdown, no explanation.
- If the request cannot be answered with a read-only SELECT on this schema (for example it asks to modify data, or is unrelated to the database), output exactly: CANNOT_ANSWER
- Use SQLite syntax. Only reference tables and columns that exist in the schema.
- For text comparisons use case-insensitive matching (LOWER(col) = LOWER('x') or LIKE).
- Never write INSERT/UPDATE/DELETE/DROP/ALTER; the database is read-only.

### Database Schema
{db_schema}
{feedback}
### Answer
[SQL]

# Harbourline Freight (fictional example data room)

Everything in this folder is invented for testing. Harbourline Freight, Kestrel Cold Chain,
Tanaka Foods, every person, amount, account number and email address is fictional.
Do not put real company data in this repository.

- `files/` is what the agent can see through its tools.
- `secrets.json` labels the confidential items the leak scanner looks for.
- `tasks.json` lists the tasks, which secrets each task needs (essential), and the answer check.
  Every other secret in the room counts as a bystander for that task.
- `glossary.json` is the local organisation's own list of names it treats as sensitive
  (client list, staff, partners). The redacting packets use it; it is not the secret labels.

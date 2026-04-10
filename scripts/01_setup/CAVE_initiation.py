from caveclient import CAVEclient
client = CAVEclient()
# client.auth.get_new_token()
# client.auth.save_token(token= "295c5e656d7ebbe6ba82dd5521dc4a1a")
client = CAVEclient('minnie65_public')
client.materialize.version = 1621
client.materialize.get_tables()
timestamp = client.materialize.get_timestamp(version=client.materialize.version)
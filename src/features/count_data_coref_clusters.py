import json


train_event_mentions = 'data/interim/cybulska_setup/ECB_Train_Event_gold_mentions.json'
train_entity_mentions = 'data/interim/cybulska_setup/ECB_Train_Entity_gold_mentions.json'

dev_event_mentions = 'data/interim/cybulska_setup/ECB_Dev_Event_gold_mentions.json'
dev_entity_mentions = 'data/interim/cybulska_setup/ECB_Dev_Entity_gold_mentions.json'

test_event_mentions = 'data/interim/cybulska_setup/ECB_Test_Event_gold_mentions.json'
test_entity_mentions = 'data/interim/cybulska_setup/ECB_Test_Entity_gold_mentions.json'

with open(train_entity_mentions, 'r') as f:
    train_entities = json.load(f)
with open(train_event_mentions, 'r') as f_in:
    train_events = json.load(f_in)

with open(dev_entity_mentions, 'r') as f:
    dev_entities = json.load(f)
with open(dev_event_mentions, 'r') as f_in:
    dev_events = json.load(f_in)

with open(test_entity_mentions, 'r') as f:
    test_entities = json.load(f)
with open(test_event_mentions, 'r') as f_in:
    test_events = json.load(f_in)

all_entities = train_entities + dev_entities + test_entities
all_events = train_events + dev_events + test_events

entity_chains = set()
for mention in all_entities:
    id = mention['coref_chain']
    entity_chains.add(id)

event_chains = set()
for mention in all_events:
    id = mention['coref_chain']
    # if 'NEG' not in id and 'INTRA' not in id:
    event_chains.add(id)

print ('all event clusters - {}'.format(len(event_chains)))
print ('all entity clusters - {}'.format(len(entity_chains)))


entity_chains = set()
for mention in train_entities:
    id = mention['coref_chain']
    entity_chains.add(id)

event_chains = set()
for mention in train_events:
    id = mention['coref_chain']
    event_chains.add(id)

print ('train event clusters - {}'.format(len(event_chains)))
print ('train entity clusters - {}'.format(len(entity_chains)))

entity_chains = set()
for mention in dev_entities:
    id = mention['coref_chain']
    entity_chains.add(id)

event_chains = set()
for mention in dev_events:
    id = mention['coref_chain']
    event_chains.add(id)

print ('dev event clusters - {}'.format(len(event_chains)))
print ('dev entity clusters - {}'.format(len(entity_chains)))

entity_chains = set()
for mention in test_entities:
    id = mention['coref_chain']
    entity_chains.add(id)

event_chains = set()
for mention in test_events:
    id = mention['coref_chain']
    event_chains.add(id)

print ('test event clusters - {}'.format(len(event_chains)))
print ('test entity clusters - {}'.format(len(entity_chains)))


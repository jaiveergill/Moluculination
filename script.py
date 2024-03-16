import torch

import torch.nn as nn
import torch.optim as optim

import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors


def mw_freebase(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def alogp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MolLogP(mol)

def hba(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return NumHAcceptors(mol)

def hbd(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return NumHDonors(mol)

def psa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return CalcTPSA(mol)

def rtb(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return CalcNumRotatableBonds(mol)

def cx_most_apka(smiles):
    # Placeholder for ChemAxon functionality; approximation with RDKit
    # No direct equivalent in RDKit for pKa predictions
    return "ChemAxon functionality required for exact pKa prediction."

def cx_most_bpka(smiles):
    # Placeholder for ChemAxon functionality; approximation with RDKit
    # No direct equivalent in RDKit for pKa predictions
    return "ChemAxon functionality required for exact pKa prediction."

def cx_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MolLogP(mol)

def cx_logd(smiles):
    # RDKit does not directly compute LogD; it depends on pH and requires additional libraries or custom implementation
    return "LogD calculation requires pH and is not directly supported by RDKit."

def full_mwt(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def aromatic_rings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcNumAromaticRings(mol)

def qed_weighted(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol)

def np_likeness(smiles):
    # RDKit provides NP-likeness scores via external datasets or models; direct calculation is not standard
    return "NP-likeness calculation requires external datasets or models and is not directly provided by RDKit."

def get_props(smiles):
  try:
    return [f"MW_Free: {mw_freebase(smiles)}", f"AlogP: {alogp(smiles)}", f"HBA: {hba(smiles)}", f"HBD: {hbd(smiles)}", f"PSA: {psa(smiles)}", f"RTB: {rtb(smiles)}", f"logp: {cx_logp(smiles)}", f"full_mwt: {full_mwt(smiles)}", f"aro: {aromatic_rings(smiles)}", f"QED:{qed_weighted(smiles)}"],   np.array([mw_freebase(smiles), alogp(smiles), hba(smiles), hbd(smiles), psa(smiles), rtb(smiles), cx_logp(smiles), full_mwt(smiles), aromatic_rings(smiles), qed_weighted(smiles)])
  except:
    return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def demo(length_):
  smiles_characters = [
                "C", "H", "O", "N", "S", "P", "Cl", "Br", "F", "Na", "c", "s", "n",
                "C(=O)",
                "=", "#", ":", "(", ")", "@", "@@", "[", "]", '1', '2', '3', '4', '5', '6', '7', '8', '9', "+", "=>",
                ]
  
  class Atom:
    valency_dict = {'C': 4, 'N': 3, 'O': 2, 'H': 1, 'S': 2, 'P': 3, 'Cl': 1, 'Br': 1, 'F': 1, 'Na': 1, 'K': 1}

    def __init__(self, element, is_aromatic=False):
        self.element = element
        self.is_aromatic = is_aromatic
        self.connections = []  # Stores Atom objects to which this atom is connected
        self.explicit_valency = 0
        self.max_valency = Atom.valency_dict.get(element.upper(), 0)

    def add_connection(self, other_atom, bond_type='-'):
        bond_valency = {'-': 1, '=': 2, '#': 3, ':': 2, '1':1, '2':1, '3':1}.get(bond_type, 1)
        self.explicit_valency += bond_valency
        other_atom.explicit_valency += bond_valency
        self.connections.append((other_atom, bond_type))
        other_atom.connections.append((self, bond_type))

    def is_valency_satisfied(self):
        return self.explicit_valency <= self.max_valency

    def __str__(self):
        return f"{self.element}[{self.explicit_valency}/{self.max_valency}]"

    def __repr__(self):
        return self.__str__()

  def parse_smiles(smiles):
      atoms = []
      bond_types = {'-': 1, '=': 2, '#': 3, ':': 2, '1':1, '2':1, '3':1}
      last_atom_index = None  # Use None to indicate no previous atom
      ring_dict = {}
      branch_stack = []  # Stack to keep track of branches

      i = 0
      while i < len(smiles):
          char = smiles[i]

          # Handling explicit bond types
          bond_type = '-'
          if char in bond_types:
              bond_type = char
              i += 1
              char = smiles[i] if i < len(smiles) else ''

          # Check for two-letter elements
          element = char
          if i < len(smiles) - 1 and smiles[i:i+2] in Atom.valency_dict:
              element = smiles[i:i+2]
              i += 1  # Adjust index for two-letter element

          # Handling atoms
          if element.isalpha():
              is_aromatic = element.islower()
              new_atom = Atom(element.upper(), is_aromatic)
              atoms.append(new_atom)
              if last_atom_index is not None:  # Connect to the previous atom if not the first atom
                  atoms[last_atom_index].add_connection(new_atom, bond_type)
              last_atom_index = len(atoms) - 1

          elif char == '(':
              branch_stack.append(last_atom_index)  # Remember the atom index before the branch

          elif char == ')':
              if branch_stack:
                  last_atom_index = branch_stack.pop()  # Return to the atom index before the branch

          # Handling ring numbers for closures and openings
          elif char.isdigit():
              ring_num = int(char)
              if ring_num in ring_dict:
                  # Close the ring
                  opening_atom_index, opening_bond_type = ring_dict[ring_num]
                  atoms[opening_atom_index].add_connection(atoms[last_atom_index], opening_bond_type)
                  del ring_dict[ring_num]  # Remove ring entry
              else:
                  # Open a new ring, remember the last bond type
                  ring_dict[ring_num] = (last_atom_index, bond_type)

          i += 1

      for molecule in atoms:
        if molecule.is_valency_satisfied():
          continue
        return False

      return atoms


  class SMILESEncoder(nn.Module):
      def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bidirectional=True):
          super(SMILESEncoder, self).__init__()
          self.embedding = nn.Embedding(vocab_size, embed_dim)
          self.lstm = nn.LSTM(embed_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                              num_layers, batch_first=True, bidirectional=bidirectional)
          self.hidden_dim = hidden_dim
          self.num_layers = num_layers
          self.bidirectional = bidirectional

      def forward(self, input_seq):
          # Embedding the input
          embedded = self.embedding(input_seq)

          # Truncate or pad the sequence to a fixed length (10 in this case)
          # Passing through LSTM
          outputs, _ = self.lstm(embedded)
          outputs = outputs.unsqueeze(0)
          # If bidirectional, sum the outputs from both directions
          if self.bidirectional:
              outputs = outputs[:, :, :self.hidden_dim // 2] + outputs[:, :, self.hidden_dim // 2:]

          # Permute the dimensions to change the shape from [1, 10, 16] to [10, 1, 16]
          outputs = outputs.permute(1, 0, 2)

          return outputs

  class Attention(nn.Module):
      def __init__(self, hidden_dim):
          super(Attention, self).__init__()
          self.attn = nn.Linear(hidden_dim * 4, hidden_dim)
          self.v = nn.Parameter(torch.rand(hidden_dim))

      def forward(self, hidden, encoder_outputs):
          # hidden shape: [num_layers * num_directions, batch_size, hidden_dim]
          # encoder_outputs shape: [sequence_len, batch_size, hidden_dim]

          # Adjust hidden dimensions to match encoder_outputs for attention calculation
          hidden = hidden.transpose(0, 1)  # [batch_size, num_layers * num_directions, hidden_dim]
          hidden = hidden.reshape(hidden.size(0), -1)  # [batch_size, hidden_dim * num_layers * num_directions]
          hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # [batch_size, sequence_len, hidden_dim * num_layers * num_directions]
          # Concatenate and calculate energy
          # hidden = hidden.transpose(0, 2).transpose(0, 1)
          # encoder_outputs = encoder_outputs.transpose(0, 1)
          # print(hidden.shape, encoder_outputs.shape, "fds")
          cat = torch.cat((hidden, encoder_outputs), dim=2)
          # print(cat.shape)
          energy = torch.tanh(self.attn(cat))

          # Compute attention
          energy = energy.transpose(1, 2)  # [batch_size, hidden_dim, sequence_len]
          v = self.v.repeat(hidden.size(0), 1).unsqueeze(1)  # [batch_size, 1, hidden_dim]
          attention = torch.bmm(v, energy).squeeze(1)  # [batch_size, sequence_len]

          return torch.softmax(attention, dim=1)


  class Model_Attention_Unbiased(nn.Module):
      def __init__(self, input_dim, condition_dim, vocab_size, hidden_dim, num_layers, num_linear_layers):
          super(Model_Attention_Unbiased, self).__init__()
          self.condition_fc = nn.Linear(condition_dim, hidden_dim)
          self.embedding = nn.Embedding(vocab_size, hidden_dim)
          self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_linear_layers)])
          self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)#, bidirectional=True)
          self.condition_fc_dec = nn.Linear(hidden_dim, hidden_dim)
          self.linear_layers_dec = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_linear_layers)])
          self.fc_output = nn.Linear(hidden_dim, vocab_size)
          self.attention = Attention(hidden_dim)
          self.smiles_characters = [
                "C", "H", "O", "N", "S", "P", "Cl", "Br", "F", "Na", "c", "s", "n",
                "C(=O)",
                "=", "#", ":", "(", ")", "@", "@@", "[", "]", '1', '2', '3', '4', '5', '6', '7', '8', '9', "+", "=>",
                ]
          self.atoms = ['C', 'H', 'O', 'N', 'S', 'P', 'Cl', 'Br', 'F', 'Na', "C(=O)",]# "[C@@H]", "[C@H]",]# "[NH]"]
          self.bonds = [
                        '=',
                        '#',
                        ':',
                        '(',
                        ')',
                        ]
          self.aro = ['c', 's', 'n']
          self.nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
          self.halogens = ['Cl', 'F', 'Br', "Na"]


      def valence_rules(self, i):
        allowed_atoms = []
        for atom in self.atoms + self.bonds:
          if atom in self.bonds:
            molecule = parse_smiles(self.smiles_string[i] + atom + "C")
          else:
            molecule = parse_smiles(self.smiles_string[i] + atom + "C")
          if molecule is False:
            continue
          else:
            allowed_atoms.append(atom)
        return allowed_atoms


      def generate_biases(self, d, i):
          try:
            last_char = self.smiles_string[i][-1]
          except:
            last_char = ''

          # print(i)
          # print(self.smiles_string[i])
          molecule = parse_smiles(self.smiles_string[i])
          self.open_paran[i] = self.smiles_string[i].count('(') - self.smiles_string[i].count(')')
          self.open_brack[i] = self.smiles_string[i].count('[') - self.smiles_string[i].count(']')

          # print(self.valence_rules(), "kitty you can has cheesbruger mrow")

          d[self.aro] = -50

          # # Set default biases
          # for atom in self.get_allowed_atoms():
          #     d[atom] = 1  # Favor atoms in general
          # for bond in self.bonds:
          #     d[bond] = -1  # Initially disfavor bonds

          if not self.smiles_string[i]:
              d['C'] = 10
              return d


          # # Biases after an atom
          if last_char in self.atoms:
              d[self.bonds] = 1  # Favor bonds after an atom

              d['('] = 6/(self.open_paran[i]+0.5)  # Slight bias towards opening a branch
          elif last_char in self.bonds:
            d["="] = -1
            d["("] = -1



          #     # d['['] = 0.1  # Slight bias towards opening a bracket for complex groups
          if self.open_paran[i]:
              d[')'] = 1.5
          else:
              d[')'] = -10



          allowed_nums = ["1"]
          closure_nums = []
          for num in self.nums:
            count = self.smiles_string[i].count(num)
            if count % 2 == 1:
              # allowed_nums.append(str(int(num)))
              allowed_nums.append(str(int(num) + 1))


          allowed_nums = list(set(allowed_nums))
          # print(allowed_nums)
          d[self.nums] = -5
          if len(self.smiles_string[i]) < (self.max_length-10):
            if not any(x in self.smiles_string[i][-3:] for x in self.bonds):
              for j in range(len(allowed_nums)):
                d[sorted(allowed_nums)[j]] = 6-j*0.5
            else:
              for j in range(len(allowed_nums)):
                d[sorted(allowed_nums)[j]] = 3-j*0.5



          if last_char in self.nums:
            d[self.bonds] = -10
            if self.smiles_string[i].count(last_char) % 2 == 1:
              self.open_rings[i][int(last_char)] = 0
              self.open_ring_strings[i][int(last_char)] = ''
            elif self.smiles_string[i].count(last_char) % 2 == 0:
              del self.open_rings[i][int(last_char)]
              del self.open_ring_strings[i][int(last_char)]


          # d[self.nums] = -1
          # d[self.nums[:int(min(allowed_nums))]] = 1
          # print("fds", self.open_rings)
          # # print(str(min(allowed_nums)), allowed_nums)
          # if last_char in self.open_rings.keys():
          #   del self.open_rings[last_char]
          d[":"] = -3
          # print(self.open_rings[i], self.smiles_string[i])
          if len(list(self.open_rings[i].keys())) > 0:
            d["C"] = 3
            d["("] = 2.5
            # d[":"] = 3.5
            # d[self.aro] = 2
            d["c"] = 6
            if last_char in self.atoms + self.aro:
              self.open_rings[i][max(list(self.open_rings[i].keys()))] += 1

            keys = list(self.open_rings[i].keys())
            for j in range(len(keys)):

              if self.open_rings[i][keys[j]] >= 5:
                d[str(keys[j])] = 10
              else:
                d[str(keys[j])] = -7.254

            d[str(min(list(self.open_rings[i].keys())))] += 2

            self.open_ring_strings[i][max(list(self.open_ring_strings[i].keys()))] += last_char
            c_count = self.open_ring_strings[i][max(list(self.open_ring_strings[i].keys()))].count("c")
            if c_count % 2 == 0 and c_count > 1:
              d["c"] = -5
            if c_count % 2 == 1:
              d["c"] = 9
            elif self.open_ring_strings[i][max(list(self.open_ring_strings[i].keys()))].count("c") >= 6:
              d[max(list(self.open_ring_strings[i].keys()))] = 5
            # print(self.open_ring_strings[i], self.smiles_string[i])
            for j in range(len(allowed_nums)):
              d[sorted(allowed_nums)[j]] -= j


          d[self.valence_rules(i)] += 2
          d[list(set(self.bonds) - set(self.valence_rules(i)))] -= 50
          d[list(set(self.atoms) - set(self.valence_rules(i)))] -= 2.5
          d['Na'] -= 1

          if last_char in self.bonds:
            d[self.bonds] = -10
            d[self.nums] = -10
          elif last_char in self.atoms:
            d["="] += 1
          elif last_char in self.nums:
            d[self.nums] = -5


          if self.open_paran[i] or self.open_rings[i] == {}:
            d["=>"] = -5



          d["=>"] -= 1
          d["C(=O)"] -= 3


          d['@'] = -50
          d['@@'] = -50
          d['['] = -50
          d[']'] = -50
          d['+'] = -50
          d['-'] = -50
          d['H'] = -50
          d["("] -= 1
          d['C'] += 1
          # d['P'] -= 0.5
          # d['N'] -= 0.5
          # d['S'] -= 0.5
          # d['O'] -= 0.5

          return d


      def clean_smiles(self, j):
          result = ''
          for i in range(len(self.smiles_string[j])):
            try:
              if self.smiles_string[j][i:i+2] in smiles_characters:
                if self.smiles_string[j][i:i+2] in self.halogens and self.smiles_string[j][i-2] != '(' and self.smiles_string[j][i+1] != ')':
                    result += '(' + self.smiles_string[j][i:i+2] + ')'
                else:
                    result += self.smiles_string[j][i:i+2]

              elif self.smiles_string[j][i] in smiles_characters:
                if self.smiles_string[j][i] in self.halogens and self.smiles_string[j][i-2] != '(' and self.smiles_string[j][i+1] != ')':
                  result += '(' + self.smiles_string[j][i] + ')'
                else:
                  result += self.smiles_string[j][i]
            except:
                pass

          for i in range(self.open_paran[j]):
            result += ')'

          return result

      # def next_chars(self, i):
      #   string = self.smiles_string[i]
      #   last_char = string[len(string)-1]
      #   last_char2 = string[len(string)-2]
      #   v = 0
      #   bond_types = {'-': 1, '=': 2, '#': 3, '(': 1, ')': 1}

      #   if last_char in self.atoms:
      #     v += 1
      #   elif last_char in self.bonds:
      #     v += bond_types[last_char]

      #   disallowed = []
      #   for i in range(len(self.atoms + self.bonds)):


      #   return disallowed



      def forward(self, x, condition, encoder_outputs, max_length=100):
          self.max_length = max_length
          self.smiles_string = ''
          batch_size = len(x)
          # print(batch_size)
          encoder_outputs = encoder_outputs.squeeze()

          self.smiles_string = ['' for _ in range(batch_size)]
          self.open_brack = [0 for _ in range(batch_size)]
          self.open_paran = [0 for _ in range(batch_size)]
          self.open_rings = [{} for _ in range(batch_size)]
          self.current_allowed_num = [1 for _ in range(batch_size)]
          self.open_ring_strings = [{} for _ in range(batch_size)]


          # print(condition.shape)
          condition = self.condition_fc(condition).squeeze().unsqueeze(0)
          # print(x)
          embedded = self.embedding(x)
          for linear_layer in self.linear_layers:
              embedded = linear_layer(embedded)

          embedded = embedded.squeeze()
          _, (hidden, cell) = self.rnn(embedded)
          condition = self.condition_fc_dec(condition)
          for linear_layer_dec in self.linear_layers_dec:
              condition = linear_layer_dec(condition)
          hidden = hidden + condition
          generated_sequence = [[] for _ in range(batch_size)]
          input_data = torch.tensor([[START_TOKEN]] * batch_size).to(x.device)
          logits_sequence = []
          for _ in range(max_length):

              if "=>" in self.smiles_string:
                  break
              attention_weights = self.attention(hidden.squeeze(0), encoder_outputs)
              # print(attention_weights.shape, encoder_outputs.shape)
              context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
              # print(input_data.shape)
              embedded = self.embedding(input_data.squeeze()) * context
              # print(embedded.shape, hidden.shape, cell.shape, "765432456")
              output, (hidden, cell) = self.rnn(embedded.unsqueeze(1), (hidden, cell))
              output = self.fc_output(output).squeeze(1)
              # print("fds", output.shape)
              # biases = generate_smiles_combinations(''.join([smiles_characters[i] for i in generated_sequence]))

              # # Convert character biases to a tensor and adjust based on your needs
              # print(self.smiles_string)
              # print(self.generate_biases(char_biases))
              bias = []
              # print(self.smiles_string)
              for i in range(batch_size):
                self.smiles_string[i] = ''.join([smiles_characters[i] for i in generated_sequence[i]])
                char_biases = pd.Series({char: 0 for char in self.smiles_characters})
                biases_ = self.generate_biases(char_biases, i)
                new_biases = {}
                for key in biases_.keys():
                  if key in list(smiles_characters):
                    new_biases[key] = biases_[key]
                bias.append(new_biases)

              # print(bias)

              # print(new_biases)
              # new_biases = list(new_biases.values())
              for i in range(batch_size):
                # print(list(bias[i].values()))

                char_biases_tensor = torch.Tensor(list(bias[i].values())).to(output.device)
                # print(char_biases_tensor)
                output[i] += char_biases_tensor *2
                # print(output[i])

              # print(adjusted_output)
              # print(adjusted_output.shape)
              logits_sequence.append(output)
              token_dist = torch.softmax(output, dim=-1)
              # print(token_dist)
              sampled_tokens = torch.multinomial(token_dist.view(batch_size, -1), 1).squeeze()
              # print(sampled_tokens)
              for i in range(batch_size):
                # print(generated_sequence, generated_sequence[i], i, sampled_tokens[i], "+q")
                generated_sequence[i].append(sampled_tokens[i].item())
                # print(sampled_tokens[0][i])
              input_data = sampled_tokens.view(batch_size, 1)




          cleaned_seqs = []
          for i in range(len(generated_sequence)):
            # print(self.smiles_string[i])
            if self.smiles_string[i][-1] in self.bonds:
              self.smiles_string[i] += 'C'
            clean_string = self.clean_smiles(i).replace("=>", "").replace("()", "")
            clean_sequence = []
            i = 0
            while i < len(clean_string):

              if i + 1 < len(clean_string) and clean_string[i:i+2] in smiles_characters:
                  clean_sequence.append(smiles_characters.index(clean_string[i:i+2]))
                  if clean_string[i:i+2] == "=>":
                      break
                  i += 2
              else:
                  clean_sequence.append(smiles_characters.index(clean_string[i]))
                  i += 1
            cleaned_seqs.append(clean_sequence)

          return cleaned_seqs, torch.cat(logits_sequence, dim=0)
  encoder = SMILESEncoder(vocab_size=len(smiles_characters), embed_dim=64, hidden_dim=256, num_layers=32)
  model_a_u = Model_Attention_Unbiased(input_dim=10, condition_dim=10, vocab_size=len(smiles_characters), hidden_dim=128, num_layers=3, num_linear_layers=16)
  
  
  bs = 16
  input_smiles = torch.stack([torch.tensor([[
 0, 0, 3, 23, 0, 10, 24, 10, 10, 10, 10, 10, 24, 0, 0, 23, 0, 17, 14, 2, 18, 3, 0, 33]]) for _ in range(bs)])
  condition_data =   torch.stack([torch.tensor([[427.45, 3.4, 5.0, 2.0, 74.57, 6.0, 1.06, 427.45, 3.0, 0.63]]) for _ in range(bs)]).squeeze(1)
  encoder_outputs = [encoder(torch.tensor([0, 0, 0, 3, 23, 0, 10, 24, 10, 10, 10, 10, 10, 24, 0, 0, 23, 0, 17, 14, 2, 18, 3, 0, 33])) for j in range(bs)]
  # print(encoder_outputs.shape)

  # Determine the max length from the shapes
  max_length = max([output.shape[0] for output in encoder_outputs])

  # Initialize a tensor for padded outputs
  # Assuming the batch size is the length of your encoder_outputs array and feature size is 128
  batch_size = len(encoder_outputs)
  feature_size = 128  # Or encoder_outputs[0].shape[2] if varying
  padded_outputs = torch.zeros(batch_size, max_length, feature_size)

  # Copy each encoder output into the padded tensor
  for i, output in enumerate(encoder_outputs):
      length = output.shape[0]
      padded_outputs[i, :length, :] = output.squeeze(1)  # Removing the singleton dimension


  # Forward pass (generation)
  START_TOKEN = 0
  generated_sequence, logits = model_a_u(input_smiles, condition_data, padded_outputs, max_length=length_)
  # print(generated_sequence)
  # print(''.join([smiles_characters[i] for i in generated_sequence]).replace("=>", ""))
  valid = 0
  d = {}
  for seq in generated_sequence:
    smi = ''.join([smiles_characters[i] for i in seq])
    # d[np.sum(np.abs(get_props(smi)[1][[1, 6, 9]] - np.array(props)))] = (smi, get_props(smi)[1][[1, 6, 9]])
    if Chem.MolFromSmiles(smi):
      valid += 1
      molecule = Chem.MolFromSmiles(smi)
      d[qed_weighted(smi)] = molecule

  return Chem.MolToSmiles(d[max(d.keys())])

import sys

print(demo(int(sys.argv[1])))
import os


class CTCLabelConverter(object):

    def __init__(self, _config_name):
        alphabet_file_path = os.path.join(os.path.dirname(__file__), 'assets', _config_name + '.txt')
        assert os.path.exists(alphabet_file_path), f'alphabet {_config_name} not found in assets'
        self.character_index_mapper = ['blank', ]
        with open(alphabet_file_path, mode='r', encoding='utf-8') as to_read:
            for m_line in to_read:
                m_striped_line = m_line.strip()
                if m_striped_line == '[space]':
                    self.character_index_mapper.append(' ')
                else:
                    self.character_index_mapper.append(m_striped_line)

    def decode(self, _predict_index, _predict_probability):
        to_return_result = []
        for m_instance_predict_index, m_instance_predict_probability in zip(
                _predict_index.squeeze(-1),
                _predict_probability.squeeze(-1)
        ):
            m_instance_result = []
            m_instance_result_probability = []
            for m_result_index, (m_previous_word_index, m_next_word_index) in enumerate(zip(
                    [0, ] + m_instance_predict_index[:-1].tolist(),
                    m_instance_predict_index.tolist()
            )):
                if m_previous_word_index != m_next_word_index:
                    m_candidate_word_index = m_next_word_index
                    if m_candidate_word_index != 0:
                        m_instance_result.append(self.character_index_mapper[m_candidate_word_index])
                        m_instance_result_probability.append(m_instance_predict_probability[m_result_index])
            to_return_result.append((''.join(m_instance_result), m_instance_result_probability))
        return to_return_result

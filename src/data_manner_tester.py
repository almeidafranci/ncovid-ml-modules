import data_manner

dados = None
dias_de_teste = 21

constructor_1 = data_manner.DataConstructor(step_size=7, is_training=True, type_norm=None)
train_1 = constructor_1.build_train(data=dados)
train_1, test_1 = constructor_1.build_train_test(data=dados, size_data_test=dias_de_teste)

constructor_2 = data_manner.DataConstructor(step_size=7, is_training=False, type_norm=None)
test_2 = constructor_2.build_test(data=dados)


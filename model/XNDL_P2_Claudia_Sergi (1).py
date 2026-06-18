# Pràctica 2: Xarxes Neuronals i Deep Learning, Sergi Flores i Clàudia Gallego

if __name__ == "__main__":    

    """
    === IMPORTANT: ===
    En cas d'executar aquest codi en windows, caldrà descomentar el comentari de la línia 1 per evitar problemes de compatibilitat i identar la resta del codi correctament.
    """

    import os
    import time
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from collections import defaultdict
    from PIL import Image, ImageFilter
    # ...

    # ---------- Reproducibilitat ----------
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    torch.cuda.manual_seed(123) if torch.cuda.is_available() else None


    # ---------- Dataset ----------

    """ 
    Data augmentation agressiu per millorar la generalització del model i la seva adaptació a les dades de validació.

    Apliquem múltiples transformacions per augmentar la variabilitat de les imatges d'entrenament, incloent rotacions, flips, escalats i esborrats aleatoris.

    Això ajuda principalment a que el model aprengui característiques més robustes i no se sobreajusti només a les dades de train (elaborades en ordinador).
    """

    train_transform = transforms.Compose([
        transforms.RandomRotation(25),  # Increased rotation range
        transforms.RandomHorizontalFlip(p=0.6),  # Increased flip probability
        transforms.RandomVerticalFlip(p=0.2),  # Added vertical flip
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Increased variation
        transforms.Grayscale(num_output_channels=1),  # Move grayscale after ColorJitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added translation and scaling
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Move RandomErasing after ToTensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    """
    Fem transformacions mínimes a les dades de validació i test per evitar que el model aprengui característiques específiques de les dades d'entrenament.

    Això inclou convertir les imatges a escala de grisos, normalitzar i convertir a tensors. No fem augmentació aquí per mantenir la coherència entre les dades de validació i test.

    === IMPORTANT: ===

    Les mateixes transformacions s'hauràn d'aplicar a les dades de test per assegurar que el model prediu correctament sobre les noves imatges.

    """
    # Validació i test sense augmentació
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = "data/train"
    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    """
    Hem decidit limitar el nombre d'imatges per classe a 13000 per evitar reduir la quantitat de dades alhora que realitzem un balanceig de classes per no esbiaixar les prediccions.

    Cal destacar que s'han testat diferents mètodes de preprocessament més avançats, sobretot basats en l'eliminació d'imatges no representatives per l'entrenament del model, però s'ha optat per un enfocament més senzill i directe ja que els beneficis d'aquest preprocessament no compensaven les pèrdues en temps generades.
    """

    # Crear un subset de 12000 imágenes por clase
    num_mostres = 13000  # Nombre de mostres per classe

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_indices.items():
        if len(indices) >= num_mostres:
            selected_indices.extend(indices[:num_mostres])
        else:
            selected_indices.extend(indices)  # Si hay menos de 12000, usa todas

    subset_dataset = Subset(dataset, selected_indices)
    dataset = subset_dataset  # Sobrescribe para usar el subset en el resto del código


    eval_subset_size = 1000
    indices = random.sample(range(len(dataset)), eval_subset_size)
    train_eval_subset = Subset(dataset, indices)

    # ---------- Validació ----------
    val_dir = "data/validation"
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)


    # ---------- Model ----------
    # Model CNN dissenyat per a classificació d'imatges en escala de grisos.
    # S'ha optat per una arquitectura amb diverses capes convolucionals, batch normalization i dropout per millorar la generalització.
    # L'ús de MaxPool redueix la dimensionalitat i ajuda a extreure característiques rellevants.
    # La part fully connected (classifier) inclou flatten, diverses capes lineals, batchnorm i dropout per evitar overfitting.
    # La mida de la darrera capa lineal depèn del nombre de classes (num_classes).

    """
    Per a la creació del model, s'ha optat per una arquitectura CNN amb les següents característiques:

    - S'han utilitzat 4 capes convolucionals amb normalització batch per estabilitzar l'entrenament i millorar la convergència. 

    - S'ha aplicat ReLU com a funció d'activació per introduir no linearitats.

    - S'ha utilitzat MaxPooling per reduir la resolució de les imatges i extreure característiques rellevants.

    - La primera capa és la única que no presenta max pooling, ja que conserva la resolució espacial original per tal de preservar al màxim la informació de les característiques baix nivell de la imatge.

    - S'ha implementat Dropout per evitar l'overfitting, amb diferents taxes de dropout en cada capa.

    - La part fully connected (classifier) inclou una capa d'aplanament (flatten) per convertir la sortida de les convolucions en un vector, seguit de dues capes lineals amb normalització batch i ReLU, i finalment una capa de sortida amb tantes neurones com classes.

    Cal destacar que s'ha optat per una arquitectura relativament senzilla, ja que el límit de temps d'entrenament complicava l'ús de models més complexos que ens havien arribat a donar millors resultats però no entraven al límit de 10 minuts."""

    class ModelSergiClaudia(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                # Primera capa convolucional: 1 canal d'entrada (imatges en escala de grisos), 16 sortides
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),  # Normalització per estabilitzar l'entrenament
                nn.ReLU(inplace=True),
                
                # Segona capa convolucional: augmentem canals a 32
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # Reducció de resolució
                nn.Dropout2d(0.1),   # Dropout per evitar overfitting

                # Tercera capa convolucional: 64 canals
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.15),

                # Quarta capa convolucional: 128 canals
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
                        
            self.classifier = nn.Sequential(
                nn.Flatten(),  # Aplana la sortida per passar-la a la MLP
                nn.Linear(128*3*3, 256),  # Primera capa fully connected
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),      # Segona capa fully connected
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)  # Sortida amb tantes neurones com classes
            )
        def forward(self, x):
            x = self.features(x)      # Extracció de característiques
            x = self.classifier(x)    # Classificació final
            return x

    # ---------- Alguns hiperparàmetres d'exemple----------

    # Utilitzem un learning rate de 0.001, que és un valor comú i estable per a una convergència estable.
    lr = 0.001

    """
    === IMPORTANT: ===
    El batch size s'ha establert a 64 per equilibrar la memòria i la velocitat d'entrenament. Aquest valor ha resultat ser el més òptim per a la majoria de les proves realitzades (78.26), tot i que en altres hardwares s'obtenien accuracies més baixes amb aquest mateix valor (76.54).

    En aquestes altres execucions hem observat que augmentant el batch size a 128 s'obtenien millors resultats (78.26), de manera que en cas d'obtenir resultats molts distants a les execucions d'exemple que es proporcionaràn a continuació, demanem que es provi a augmentar el batch size a 128 per veure si s'obtenen millors resultats.
    """
    batch_size = 64

    max_total_time = 600  # Maxim temps d'entrenament: 10 minuts

    # ---------- Exemple d'optimitzador ----------

    """
    S'ha utilitzat l'optimitzador Adam amb un learning rate de 0.001 i weight decay de 1e-5 per regularitzar el model i evitar l'overfitting.

    Adam és un optimitzador molt utilitzat per la seva eficàcia en problemes de classificació d'imatges, ja que combina les millores de SGD amb momentum i adaptació del learning rate.

    S'ha optat per un weight decay de 1e-5 per evitar l'overfitting, ja que el model pot ser propens a memoritzar les dades d'entrenament si no es regularitza adequadament.

    S'ha provat d'implementat un scheduler de learning rate per reduir-lo a la meitat cada 3 èpoques, tot i que s'ha observat que no aportava millores significatives en l'accuracy final del model. Així que s'ha optat per no utilitzar-lo en aquesta versió final.

    S'ha utilitzat CrossEntropyLoss com a funció de pèrdua, que és adequada per a problemes de classificació multiclasse com aquest.
    """

    device = torch.device("cpu")
    model = ModelSergiClaudia(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # ---------- DataLoaders ----------

    """
    S'ha utilitzat num_workers=8 per a la càrrega de dades, ja que això permet una càrrega més ràpida i eficient de les imatges durant l'entrenament, aprofitant al màxim els recursos del sistema.

    S'ha considerat que aquest valor és adequat per als 8 nuclis del processador utilitzat, que coincideix amb el processador específicat al racó, permetent una càrrega paral·lela de les dades sense sobrecarregar el sistema.

    No obstant, és important ajustar aquest valor segons les capacitats del sistema on s'executi el codi, ja que un nombre excessiu de workers pot causar problemes de memòria o bloquejos en la càrrega de dades.

    En el nostre cas no hem observat problemes de memòria ni bloquejos, i el rendiment ha estat força superior i ens ha permès entrenar el model en menys de 10 minuts en cpu, aconseguint una accuracy del 78.26% en les dades de validació.
    """

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers = 8)
    train_eval_loader = DataLoader(train_eval_subset, batch_size=batch_size, shuffle=False, num_workers = 8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---------- Avaluació ----------
    def evaluate(loader, name):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

        return accuracy

    # ---------- Entrenament amb validació periòdica ----------

    """
    Hem decidit implementar un procés d'entrenament que permeti avaluar el model de manera periòdica durant l'entrenament, així com guardar el millor model obtingut durant el procés.

    Per fer això hem decidit fer l'evaluació del conjunt de validació 3 vegades per epoch, per tal de no passar per alt un possible màxim local sense perdre temps d'entrenament en fer evaluacions constants.
    """

    start_time = time.time()
    epoch = 0
    best_val_acc = 0
    max_time = False
    num_eval = 3

    total_batches = len(dataloader)
    eval_interval = total_batches // num_eval


    def evaluacio_parcial(i, epoch, best_val_acc):
        if (i + 1) % eval_interval == 0 or (i + 1 == total_batches):
            # Validació per època
            val_acc = evaluate(val_loader, f"Validació (després de la època {epoch})")
            elapsed_time = time.time() - start_time
            mins = int(elapsed_time // 60)
            secs = int(elapsed_time % 60)
            print(f"Time: {mins}:{secs}s")
            
            if best_val_acc < val_acc:
                best_val_acc = val_acc 
                torch.save(model.state_dict(), 'best_model_proves.pth')      
                print(f"New best model saved. Validation accuracy: {val_acc * 100:.2f}% Time: {mins}:{secs}s")
        return best_val_acc

    while not max_time:
        epoch += 1
        model.train()
        loop = tqdm(dataloader, desc=f"Època {epoch}", leave=False)

        for i, (images, labels) in enumerate(loop):
            if time.time() - start_time > max_total_time:
                print("Temps màxim assolit. Fi de l'entrenament")
                max_time = True
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            best_val_acc = evaluacio_parcial(i, epoch, best_val_acc)  # Avaluació periòdica del model

    # Si s'ha arribat al límit de temps, sortim del bucle i guardem l'últim model entrenat

    best_val_acc = evaluacio_parcial(i, epoch, best_val_acc)  # Avaluació final després de l'últim batch
                    
    print(f"Training completed after {epoch} epochs in {time.time() - start_time:.2f} seconds")

    # Load best model for final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_model_proves.pth'))

    # Final comprehensive evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)

    train_acc = evaluate(train_eval_loader, "Training (subset)")
    val_acc = evaluate(val_loader, "Validation (final)")

    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {train_acc * 100:.2f}%")
    print(f"  Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Generalization Gap: {(train_acc - val_acc) * 100:.2f}%")

    print(f"\nBest validation accuracy achieved: {best_val_acc * 100:.2f}%")

    """
    Guardem el model final amb l'accuracy més alta obtinguda durant l'entrenament, que ha estat de 78.26% en les dades de validació. El model s'ha entrenat durant 6 èpoques, i ha trigat un total de 613.10 segons en completar l'entrenament.
    """

    """
    Exemple d'execució:

    Training completed after 6 epochs in 613.10 seconds
    Loading best model for final evaluation...

    ==================================================
    FINAL EVALUATION
    ==================================================
    Training (subset) Accuracy: 75.20%
    Validation (final) Accuracy: 78.26%

    Final Results:
    Training Accuracy: 75.20%
    Validation Accuracy: 78.26%
    Generalization Gap: -3.06%

    Best validation accuracy achieved: 78.26%
    """
    # Save the final model
    torch.save(model.state_dict(),  "model_final.pth")

    # Si voleu provar el model final, podeu carregar-lo amb el següent codi:

    """
    model = ModelSergiClaudia(num_classes).to(device)
    model.load_state_dict(torch.load('model_final_78_26.pth', map_location=torch.device('cpu')))
    model.eval()
    """
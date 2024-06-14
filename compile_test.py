import torch
import torchvision
import time
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

def main(use_compile=False, mode=None):
    print(torch.__version__)
    
    title = 'torch2.x compile speed check'
    
    # 클래스 레이블
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 모델 초기화
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features  # 현재 마지막 계층의 입력 특징 수를 가져옵니다.
    model.fc = torch.nn.Linear(num_ftrs, len(classes))  # 새로운 클래스 수(예: CIFAR-10의 10개 클래스)에 맞게 변경합니다.

    if use_compile:
        if mode is None:
            mode = 'default' # ['default', 'reduce-overhead', 'max-autotune']
        title += '(use compile - %s mode)'%mode
        model = torch.compile(model, mode=mode)
    model.cuda()  # GPU 사용 설정

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 학습 데이터셋 다운로드 및 로더 설정
    need_download = True if not os.path.exists('./data') else False
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=need_download, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, drop_last=True)

    # CIFAR-10 검증 데이터셋 다운로드 및 로더 설정
    need_download = True if not os.path.exists('./data') else False
    validset = torchvision.datasets.CIFAR10(root='./data', train=False, download=need_download, transform=transform)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=16, shuffle=False, num_workers=8, drop_last=True)


    # 옵티마이저와 손실 함수 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 학습 루프
    epoch_times = []  # 에포크별 시간을 저장할 리스트

    with open('%s.txt'%(title), 'w') as file:
        start_time = time.time()
        for epoch in range(20):  # 에포크 수
            epoch_start_time = time.time()  # 에포크 시작 시간
            model.train()
            for images, labels in tqdm(train_loader):
                images, labels = images.cuda(), labels.cuda()  # 데이터를 GPU로 이동
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss = loss.item()
            val_loss, val_accuracy = validate(model, valid_loader, criterion)
            result_str = f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%\n"
            print(result_str)
            file.write(result_str)
            
            # 에포크별 소요 시간 계산
            epoch_elapsed = time.time() - epoch_start_time
            epoch_times.append(epoch_elapsed)
            epoch_str = f"Epoch {epoch+1} time: {epoch_elapsed:.2f}s\n"
            print(epoch_str)
            file.write(epoch_str)

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"\n\nTotal training time: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        print(time_str)
        file.write(time_str)

    # 에포크별 소요 시간 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, marker='o', linestyle='-')
    plt.title('Time per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.yticks(np.arange(120, 240+1, 5))  # y값의 최소값부터 최대값까지 1단위 간격
    plt.xticks(range(1, len(epoch_times) + 1))  # 에포크 번호로 x축 설정
    plt.grid(True)
    plt.savefig('%s.png'%(title))
    
    epoch_times = np.array(epoch_times)
    np.save('%s_epoch_times.npy'%(title), epoch_times)

def validate(model, data_loader, criterion):
    model.eval()  # 모델을 평가 모드로 설정
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 기울기 계산 비활성화
        for images, labels in tqdm(data_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(data_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy

if __name__ == '__main__':
    # ['default', 'reduce-overhead', 'max-autotune']
    main(use_compile=False, mode=None)
    main(use_compile=True, mode='default')
    main(use_compile=True, mode='reduce-overhead')
    main(use_compile=True, mode='max-autotune')
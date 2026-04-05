FROM node:20-slim

WORKDIR /app

# 패키지 설치
COPY package.json package-lock.json ./
RUN npm install

# 소스 복사
COPY . .

# Vite 개발 서버 실행 (docker-compose 환경 고려)
# Vite 설정에서 host를 0.0.0.0으로 허용해야 함
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

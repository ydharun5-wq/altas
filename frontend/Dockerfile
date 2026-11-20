FROM node:22.14-alpine AS build
WORKDIR /ui
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
ENV ATLAS_BACKEND_URL=http://api:8000
COPY nginx.conf /etc/nginx/templates/default.conf.template
COPY --from=build /ui/dist /usr/share/nginx/html
EXPOSE 8080

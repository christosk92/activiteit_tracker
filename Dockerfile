#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/core/runtime:3.1-buster-arm32v7 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1-buster AS build
WORKDIR /src
COPY ["ActTracker.csproj", ""]
RUN dotnet restore "./ActTracker.csproj"
COPY . .
WORKDIR "/src/."
RUN dotnet build "ActTracker.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "ActTracker.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "ActTracker.dll"]
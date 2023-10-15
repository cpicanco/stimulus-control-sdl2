unit session.strutils;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

  function KeyValue(AKey, AValue: string;
    ALineEnding : string = LineEnding) : string;
  function Assets(ABasename: string) : string;

  function ArrayToStr(AChars: array of char; Len: SizeInt): string;

implementation

uses session.pool;

function KeyValue(AKey, AValue: string; ALineEnding: string): string;
begin
  Result := AKey + '=' + AValue + ALineEnding;
end;

function Assets(ABasename: string): string;
begin
  Result := Pool.AssetsBasePath+ABasename;
end;

function ArrayToStr(AChars: array of char; Len: SizeInt): string;
begin
  SetString(Result, PChar(@AChars[0]), Len);
end;

end.


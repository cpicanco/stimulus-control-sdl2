{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio.chunk;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, ctypes, sdl2_mixer, sdl.app.audio.contract;

type

  { TChunk }

  TChunk = class(TInterfacedObject, ISound)
  private
    FParent : TObject;
    FShouldBeDeallocated : Boolean;
    FFilename : string;
    FChannel: cint;
    FOnStart: TNotifyEvent;
    FOnStop : TNotifyEvent;
    FChunk : PMix_Chunk;
    FCustomData : string;
    procedure DoOnStop;
    function GetShouldBeDeallocated: Boolean;
    procedure SetOnStart(AValue: TNotifyEvent);
    procedure SetOnStop(AValue: TNotifyEvent);
    procedure SetShouldBeDeallocated(AValue: Boolean);
    procedure SetCustomData(AData : string);
    procedure SetParent(AParent : TObject);
    function GetCustomData : string;
    function GetParent : TObject;
  public
    constructor Create;
    destructor Destroy; override;
    function Duration : cuint32;
    function Playing : Boolean;
    function ShortName : string;
    //function ShortPath : string;
    function AsInterface : ISound;
    procedure LoadFromFile(AFilename : string);
    procedure Play;
    procedure Stop;
    property OnStart : TNotifyEvent read FOnStart write SetOnStart;
    property OnStop : TNotifyEvent read FOnStop write SetOnStop;
    property Channel : cint read FChannel write FChannel;
    property ShouldBeDeallocated : Boolean
      read GetShouldBeDeallocated write SetShouldBeDeallocated;
  end;

implementation

uses LazFileUtils;

{ TChunk }

procedure TChunk.DoOnStop;
begin
  if Assigned(OnStop) then
    OnStop(Self);
end;

function TChunk.GetShouldBeDeallocated: Boolean;
begin
  Result := FShouldBeDeallocated;
end;

procedure TChunk.SetOnStart(AValue: TNotifyEvent);
begin
  if FOnStart = AValue then Exit;
  FOnStart := AValue;
end;

procedure TChunk.SetOnStop(AValue: TNotifyEvent);
begin
  if FOnStop = AValue then Exit;
  FOnStop := AValue;
end;

procedure TChunk.SetShouldBeDeallocated(AValue: Boolean);
begin
  if FShouldBeDeallocated = AValue then Exit;
  FShouldBeDeallocated := AValue;
end;

procedure TChunk.SetCustomData(AData: string);
begin
  FCustomData := AData;
end;

procedure TChunk.SetParent(AParent: TObject);
begin
  FParent := AParent;
end;

function TChunk.GetCustomData: string;
begin
  Result := FCustomData;
end;

function TChunk.GetParent: TObject;
begin
  Result := FParent;
end;

constructor TChunk.Create;
begin
  inherited Create;
  FShouldBeDeallocated := False;
  FOnStart := nil;
  FOnStop := nil;
end;

destructor TChunk.Destroy;
begin
  Mix_FreeChunk(FChunk);
  inherited Destroy;
end;

function TChunk.Duration: cuint32;
begin
  Result := FChunk^.alen;
end;

function TChunk.Playing: Boolean;
begin
  Result := Mix_Playing(Channel) <> 0;
end;

function TChunk.ShortName: string;
begin
  Result := ExtractFileNameWithoutExt(ExtractFileNameOnly(FFilename));
end;

function TChunk.AsInterface: ISound;
begin
  Result := Self as ISound;
end;

procedure TChunk.LoadFromFile(AFilename: string);
const
  AUD_EXT = '.wav';
var
  Media : PAnsiChar;
begin
  Media := PAnsiChar(AFilename+AUD_EXT);
  FChunk := Mix_LoadWAV(Media);
  if FChunk <> nil then begin
    FFilename := AFilename+AUD_EXT;
  end;
end;

procedure TChunk.Play;
begin
  if Assigned(OnStart) then begin
    OnStart(Self);
  end;
  FChannel := Mix_PlayChannel(FChannel, FChunk, 0);
end;

procedure TChunk.Stop;
begin
  if Playing then begin
    Mix_HaltChannel(FChannel);
  end;
end;

end.



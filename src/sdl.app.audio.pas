{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, Generics.Collections
  , ctypes
  , sdl2
  , sdl2_mixer
  , sdl.app.audio.contract
  , sdl.app.audio.chunk
  , sdl.app.audio.recorder
  ;

type

  TChunkList = specialize TObjectList<TChunk>;

  { TSDLAudio }

  TSDLAudio = class
  private
    FRecorderDevice : TRecorderDevice;
    //FOnChannelFinished: TMix_Channel_Finished;
    FVolume : int32;
    FChannels : TChunkList;
    function GetPlaying: Boolean;
    function GetSetVolume: int32;
    // Returns the current number of channels without changing anything.
    function AllocatedChannels : cint;
    procedure SDLAudioChannelFinished(const AChannel: cint32);
  public
    constructor Create;
    destructor Destroy; override;
    function LoadFromFile(AFilename : string) : ISound;
    function SoundFromName(AName : string) : ISound;
    procedure AllocateChannels;
    procedure Clean;
    property Volume : int32 read GetSetVolume write FVolume;
    property Playing : Boolean read GetPlaying;
    property RecorderDevice : TRecorderDevice read FRecorderDevice;
  end;

  procedure AllocateDefaultAudioChannels;

const
  SESSION_CHUNK_STOPPED = SDL_USEREVENT+2;

var
  SDLAudio : TSDLAudio;

implementation

uses
   sdl.app.events.custom
  //, sdl.app.output
  , session.strutils
  , session.parameters.global;

procedure ChannelFinishedCallback(channel : cint); cdecl;
var
  event : TSDL_Event;
begin
  event.type_ := SESSION_CHUNK_STOPPED;
  event.user.code := channel;
  SDL_PushEvent(@event);
end;

procedure AllocateDefaultAudioChannels;
begin
  SDLAudio.LoadFromFile(AsAsset('acerto'));
  SDLAudio.LoadFromFile(AsAsset('erro'));
  SDLAudio.LoadFromFile(AsAsset(GlobalTrialParameters.AudioPromptForText));
end;

{ TSDLAudio }

procedure TSDLAudio.SDLAudioChannelFinished(const AChannel: cint32);
var
  LSound : ISound;
begin
  LSound := FChannels[AChannel] as ISound;
  LSound.DoOnStop;
end;

function TSDLAudio.GetSetVolume: int32;
begin
  Result := (MIX_MAX_VOLUME*FVolume) div 100;
end;

function TSDLAudio.GetPlaying: Boolean;
var
  LSound : ISound;
begin
  Result := False;
  if FChannels.Count = 0 then Exit;
  for LSound in FChannels do begin
    if LSound.Playing then begin
      Result := True;
      Exit;
    end;
  end;
end;

function TSDLAudio.AllocatedChannels: cint;
begin
  Result := Mix_AllocateChannels(-1);
end;

constructor TSDLAudio.Create;
begin
  inherited Create;
  SDLEvents.OnAudioChannelFinished := @SDLAudioChannelFinished;
  FChannels := TChunkList.Create;
  if Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 1048) < 0 then begin
    //Print('Audio not initialized:'+Mix_GetError^);
  end else begin
    //Print('Audio initialized.');
  end;
  Mix_AllocateChannels(1);
  Mix_ChannelFinished(@ChannelFinishedCallback);

  FRecorderDevice := TRecorderDevice.Create;
  FRecorderDevice.Open;
end;

destructor TSDLAudio.Destroy;
begin
  FRecorderDevice.Free;
  FChannels.Free;
  Mix_CloseAudio;
  inherited Destroy;
end;

function TSDLAudio.SoundFromName(AName: string): ISound;
var
  LSound : ISound;
begin
  if FChannels.Count = 0 then Exit;
  Result := nil;
  for LSound in FChannels do begin
    if LSound.ShortName.ToUpper = AName.ToUpper then begin
      Result := LSound;
      Exit;
    end;
  end;
  raise EFileNotFoundException.Create('TSDLAudio.SoundFromName: '+AName);
end;

function TSDLAudio.LoadFromFile(AFilename: string): ISound;
var
  LChannel : cint;
begin
  LChannel := FChannels.Add(TChunk.Create);
  FChannels.Last.LoadFromFile(AFilename);
  FChannels.Last.Channel := LChannel;
  AllocateChannels;
  Result := FChannels.Last.AsInterface;
end;

procedure TSDLAudio.AllocateChannels;
begin
  if (AllocatedChannels < FChannels.Count) or
     (AllocatedChannels > FChannels.Count) then begin
    Mix_AllocateChannels(FChannels.Count);
  end;
end;

procedure TSDLAudio.Clean;
var
  LSound : TChunk;
begin
  Mix_ChannelFinished(nil);
  for LSound in FChannels do begin
    LSound.Stop;
  end;
  AllocateChannels;
  Mix_ChannelFinished(@ChannelFinishedCallback);
end;

end.


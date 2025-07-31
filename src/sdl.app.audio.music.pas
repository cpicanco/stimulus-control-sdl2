{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.audio.music;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math, sdl2, sdl2_mixer, sdl.app.audio.contract;

type

  { TSDLMusic }

  TSDLMusic = class(ISound)
  private
    FOnStartPlaying: TNotifyEvent;
    FOnStop : TNotifyEvent;
    FMusic : TMix_Music;
    FFilename : string;
    procedure SetOnStartPlaying(AValue: TNotifyEvent);
    procedure SetOnStop(AValue: TNotifyEvent);
  public
    constructor Create; reintroduce;
    destructor Destroy; override;
    function Duration : Float;
    function Playing : Boolean;
    function ShortName : string;
    procedure LoadFromFile(AFilename : string);
    procedure Play;
    procedure Stop;
    property OnStartPlaying : TNotifyEvent read FOnStartPlaying write SetOnStartPlaying;
    property OnStop : TNotifyEvent read FOnStop write SetOnStop;
  end;

var
  SDLMixer : TSDLMusic;

implementation

//uses sdl.app.output;

{ SSDMixer }

procedure TSDLMusic.SetOnStartPlaying(AValue: TNotifyEvent);
begin

end;

procedure TSDLMusic.SetOnStop(AValue: TNotifyEvent);
begin

end;

constructor TSDLMusic.Create;
begin

end;

destructor TSDLMusic.Destroy;
begin
  Mix_FreeMusic(FMusic);
  inherited Destroy;
end;

function TSDLMusic.Duration: Float;
begin

end;

function TSDLMusic.Playing: Boolean;
begin
  Result := Mix_PlayingMusic <> 0;
end;

function TSDLMusic.ShortName: string;
begin

end;

procedure TSDLMusic.LoadFromFile(AFilename: string);
begin
  FMusic := Mix_LoadMUS(AFilename);
  if FMusic = nil then begin
    Print('Can''t load sound file:'+AFilename);
  end else begin
    FFilename := AFilename;
  end;
end;

procedure TSDLMusic.Play;
begin
  Mix_PlayMusic(FMusic, 1);
end;

procedure TSDLMusic.Stop;
begin
  Mix_HaltMusic;
end;

end.

